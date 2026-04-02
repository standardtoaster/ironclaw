//! Container pool for managing warm Claude Code containers via bollard.
//!
//! One container per conversation (thread_id), multiple containers per lens.
//! Containers are created on demand, kept warm, and torn down on de-escalation
//! or explicit cleanup. IronClaw attaches to container stdin/stdout via bollard
//! for NDJSON communication.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bollard::container::{
    AttachContainerOptions, Config, CreateContainerOptions, ListContainersOptions,
    RemoveContainerOptions, StartContainerOptions,
};
use bollard::models::HostConfig;
use bollard::Docker;
use tokio::io::{AsyncBufRead, AsyncWrite, ReadBuf};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::llm::claude_protocol::{self, ClaudeStreamMessage, ExchangeResult};
use crate::llm::container_backend::ContainerBackend;
use crate::llm::error::LlmError;

/// Configuration for the container pool.
#[derive(Debug, Clone)]
pub struct ContainerPoolConfig {
    /// Docker image for Claude Code containers.
    pub image: String,
    /// Lens name (used in container naming and labels).
    pub lens: String,
    /// Claude model to use inside the container.
    pub model: String,
    /// Docker network to attach containers to.
    pub network: String,
    /// Named volume containing Claude auth credentials (mounted ro).
    pub auth_volume: String,
    /// Optional named volume for lens-specific data (mounted rw).
    pub lens_data_volume: Option<String>,
    /// Optional host path to generated/sidecar/{lens}/ config (bind mount ro).
    pub lens_config_path: Option<String>,
    /// Whether to pass --dangerously-skip-permissions to claude.
    pub skip_permissions: bool,
    /// Timeout for requests in seconds.
    pub request_timeout_secs: u64,
    /// Extra environment variables to pass to the container.
    pub extra_env: Vec<String>,
}

/// Tracks the state of an active container session for a specific thread.
struct ContainerSession {
    /// The Docker container ID.
    container_id: String,
    /// The thread UUID that owns this session (used in diagnostics/logging).
    #[allow(dead_code)]
    thread_id: Uuid,
    /// Writer for container stdin.
    stdin: Pin<Box<dyn AsyncWrite + Send + Unpin>>,
    /// Buffered line reader over container stdout.
    stdout: BollardLineReader,
    /// Number of messages sent so far (used for delta tracking).
    messages_sent: usize,
    /// CLI session ID from Claude's system/init message.
    cli_session_id: Option<String>,
}

/// A line reader that buffers NDJSON lines from a bollard attach output stream.
///
/// Bollard's attach returns a multiplexed stream of `LogOutput` items (stdout +
/// stderr). This reader filters for stdout, accumulates bytes in an internal
/// buffer, and exposes `AsyncBufRead` so `read_exchange()` can consume lines.
struct BollardLineReader {
    /// The bollard output stream (multiplexed stdout + stderr).
    stream: Pin<
        Box<
            dyn futures::Stream<Item = Result<bollard::container::LogOutput, bollard::errors::Error>>
                + Send,
        >,
    >,
    /// Internal buffer accumulating stdout bytes.
    buf: Vec<u8>,
    /// Read cursor position within `buf`.
    pos: usize,
    /// Whether the stream has ended.
    eof: bool,
}

impl BollardLineReader {
    fn new(
        stream: impl futures::Stream<Item = Result<bollard::container::LogOutput, bollard::errors::Error>>
            + Send
            + 'static,
    ) -> Self {
        Self {
            stream: Box::pin(stream),
            buf: Vec::with_capacity(4096),
            pos: 0,
            eof: false,
        }
    }
}

impl AsyncBufRead for BollardLineReader {
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<&[u8]>> {
        let this = self.get_mut();

        // If we have unconsumed data, return it.
        if this.pos < this.buf.len() {
            return Poll::Ready(Ok(&this.buf[this.pos..]));
        }

        if this.eof {
            return Poll::Ready(Ok(&[]));
        }

        // Reset buffer for new data.
        this.buf.clear();
        this.pos = 0;

        // Poll the stream for more stdout data.
        loop {
            match this.stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bollard::container::LogOutput::StdOut { message }))) => {
                    this.buf.extend_from_slice(&message);
                    return Poll::Ready(Ok(&this.buf[this.pos..]));
                }
                Poll::Ready(Some(Ok(bollard::container::LogOutput::StdErr { message }))) => {
                    // Log stderr but don't include in the reader buffer.
                    let text = String::from_utf8_lossy(&message);
                    tracing::debug!(target: "claude_container", "{}", text.trim_end());
                    // Continue polling for stdout data.
                    continue;
                }
                Poll::Ready(Some(Ok(_))) => {
                    // Console or other log output — skip.
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Err(std::io::Error::other(
                        format!("bollard stream error: {}", e),
                    )));
                }
                Poll::Ready(None) => {
                    this.eof = true;
                    return Poll::Ready(Ok(&[]));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }

    fn consume(self: Pin<&mut Self>, amt: usize) {
        let this = self.get_mut();
        this.pos += amt;
    }
}

impl tokio::io::AsyncRead for BollardLineReader {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let mut pin = Pin::new(self.get_mut());
        match pin.as_mut().poll_fill_buf(cx) {
            Poll::Ready(Ok(data)) => {
                if data.is_empty() {
                    return Poll::Ready(Ok(()));
                }
                let to_copy = std::cmp::min(data.len(), buf.remaining());
                buf.put_slice(&data[..to_copy]);
                pin.as_mut().consume(to_copy);
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Pool of warm Docker containers running Claude Code in interactive streaming mode.
///
/// Each container serves one conversation (thread_id). The pool manages creation,
/// attachment, communication, and teardown.
pub struct ContainerPool {
    docker: Docker,
    config: ContainerPoolConfig,
    sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>>,
}

impl ContainerPool {
    /// Create a new container pool connected to the Docker daemon at `socket_path`.
    ///
    /// If `socket_path` is empty, connects using bollard's defaults (DOCKER_HOST
    /// env var, then /var/run/docker.sock).
    pub async fn new(socket_path: &str, config: ContainerPoolConfig) -> Result<Self, LlmError> {
        let docker = if socket_path.is_empty() {
            Docker::connect_with_local_defaults().map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to connect to Docker: {}", e),
            })?
        } else if socket_path.starts_with("tcp://") || socket_path.starts_with("http://") {
            Docker::connect_with_http(socket_path, 120, bollard::API_DEFAULT_VERSION).map_err(
                |e| LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!("Failed to connect to Docker via TCP at {}: {}", socket_path, e),
                },
            )?
        } else {
            Docker::connect_with_socket(socket_path, 120, bollard::API_DEFAULT_VERSION).map_err(
                |e| LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!("Failed to connect to Docker at {}: {}", socket_path, e),
                },
            )?
        };

        // Verify connection.
        docker
            .ping()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Docker ping failed: {}", e),
            })?;

        tracing::info!(
            lens = %config.lens,
            image = %config.image,
            network = %config.network,
            "Container pool initialized"
        );

        Ok(Self {
            docker,
            config,
            sessions: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get or create a container session for the given thread.
    pub async fn get_or_create(&self, thread_id: Uuid) -> Result<(), LlmError> {
        // Check if session already exists.
        {
            let sessions = self.sessions.lock().await;
            if sessions.contains_key(&thread_id) {
                return Ok(());
            }
        }

        // Create a new container.
        let name = container_name(&self.config.lens, &thread_id);
        let labels = container_labels(&self.config.lens, &thread_id);

        // Build command.
        // --print (-p) is required for stream-json I/O, --verbose is required
        // for stream-json output. The process stays alive reading stdin for
        // multi-turn conversations via --input-format stream-json.
        let mut cmd = vec![
            "claude".to_string(),
            "--print".to_string(),
            "--verbose".to_string(),
            "--input-format".to_string(),
            "stream-json".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--model".to_string(),
            self.config.model.clone(),
        ];
        if self.config.skip_permissions {
            cmd.push("--dangerously-skip-permissions".to_string());
        }

        // Build environment.
        let mut env = self.config.extra_env.clone();
        env.push(format!("PERCY_LENS={}", self.config.lens));
        env.push(format!("PERCY_THREAD_ID={}", thread_id));

        // Build volume binds.
        let mut binds = vec![format!("{}:/mnt/claude-auth:ro", self.config.auth_volume)];
        if let Some(ref vol) = self.config.lens_data_volume {
            binds.push(format!("{}:/mnt/lens-data:rw", vol));
        }
        if let Some(ref path) = self.config.lens_config_path {
            binds.push(format!("{}:/mnt/lens-config:ro", path));
        }

        let host_config = HostConfig {
            binds: Some(binds),
            network_mode: Some(self.config.network.clone()),
            ..Default::default()
        };

        let container_config = Config {
            image: Some(self.config.image.clone()),
            cmd: Some(cmd),
            env: Some(env),
            labels: Some(labels),
            host_config: Some(host_config),
            open_stdin: Some(true),
            stdin_once: Some(false),
            tty: Some(false),
            ..Default::default()
        };

        let options = CreateContainerOptions {
            name: name.clone(),
            ..Default::default()
        };

        let create_response = self
            .docker
            .create_container(Some(options), container_config)
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to create container '{}': {}", name, e),
            })?;

        let container_id = create_response.id;

        // Start the container.
        self.docker
            .start_container(&container_id, None::<StartContainerOptions<String>>)
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to start container '{}': {}", name, e),
            })?;

        // Attach to stdin/stdout.
        let attach_options = AttachContainerOptions::<String> {
            stdin: Some(true),
            stdout: Some(true),
            stderr: Some(true),
            stream: Some(true),
            ..Default::default()
        };

        let attach_results = self
            .docker
            .attach_container(&container_id, Some(attach_options))
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to attach to container '{}': {}", name, e),
            })?;

        let stdin = attach_results.input;
        let mut stdout = BollardLineReader::new(attach_results.output);

        // Wait for the Claude system/init message.
        let timeout = std::time::Duration::from_secs(60);
        let init_result =
            tokio::time::timeout(timeout, wait_for_system_init(&mut stdout)).await;

        let cli_session_id = match init_result {
            Ok(Ok(sid)) => sid,
            Ok(Err(e)) => {
                // Clean up the container on failure.
                let _ = self
                    .docker
                    .remove_container(
                        &container_id,
                        Some(RemoveContainerOptions {
                            force: true,
                            ..Default::default()
                        }),
                    )
                    .await;
                return Err(e);
            }
            Err(_) => {
                let _ = self
                    .docker
                    .remove_container(
                        &container_id,
                        Some(RemoveContainerOptions {
                            force: true,
                            ..Default::default()
                        }),
                    )
                    .await;
                return Err(LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!(
                        "Timed out waiting for Claude init in container '{}'",
                        name
                    ),
                });
            }
        };

        tracing::info!(
            container_id = %container_id,
            name = %name,
            thread_id = %thread_id,
            cli_session_id = ?cli_session_id,
            "Container session created"
        );

        let session = ContainerSession {
            container_id,
            thread_id,
            stdin: Box::pin(stdin),
            stdout,
            messages_sent: 0,
            cli_session_id,
        };

        self.sessions.lock().await.insert(thread_id, session);
        Ok(())
    }

    /// Exchange a prompt with the Claude container for the given thread.
    ///
    /// The container must already exist (call `get_or_create` first).
    pub async fn exchange(
        &self,
        thread_id: Uuid,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError> {
        let timeout = std::time::Duration::from_secs(self.config.request_timeout_secs);

        let result = tokio::time::timeout(timeout, self.exchange_inner(thread_id, prompt)).await;

        match result {
            Ok(inner) => inner,
            Err(_) => {
                // Timeout — remove the session and destroy the container.
                self.remove_session(thread_id).await.ok();
                Err(LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!(
                        "Request timed out after {}s for thread {}",
                        self.config.request_timeout_secs, thread_id
                    ),
                })
            }
        }
    }

    /// Inner exchange logic: write prompt to stdin, read response from stdout.
    async fn exchange_inner(
        &self,
        thread_id: Uuid,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError> {
        use tokio::io::AsyncWriteExt;

        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(&thread_id).ok_or_else(|| {
            LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("No active session for thread {}", thread_id),
            }
        })?;

        // Write the user message as a JSON line to stdin.
        // Claude CLI stream-json format: {"message":{"role":"user","content":"..."}}
        let input = serde_json::json!({
            "message": {
                "role": "user",
                "content": prompt,
            }
        });
        let mut line =
            serde_json::to_string(&input).map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to serialize input: {}", e),
            })?;
        line.push('\n');

        session
            .stdin
            .as_mut()
            .write_all(line.as_bytes())
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to write to container stdin: {}", e),
            })?;
        session
            .stdin
            .as_mut()
            .flush()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to flush container stdin: {}", e),
            })?;

        // Read until result or approval.
        let result = claude_protocol::read_exchange(&mut session.stdout, |sid| {
            session.cli_session_id = Some(sid);
        })
        .await;

        if let Ok(ExchangeResult::Complete { .. }) = &result {
            session.messages_sent += 1;
        }

        result
    }

    /// Remove a session and destroy the associated container.
    pub async fn remove_session(&self, thread_id: Uuid) -> Result<(), LlmError> {
        let session = self.sessions.lock().await.remove(&thread_id);
        if let Some(session) = session {
            tracing::info!(
                container_id = %session.container_id,
                thread_id = %thread_id,
                "Removing container session"
            );
            self.docker
                .remove_container(
                    &session.container_id,
                    Some(RemoveContainerOptions {
                        force: true,
                        ..Default::default()
                    }),
                )
                .await
                .map_err(|e| LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!("Failed to remove container: {}", e),
                })?;
        }
        Ok(())
    }

    /// Discover existing Percy-managed containers and re-attach to them.
    ///
    /// Returns the number of sessions recovered.
    pub async fn discover_existing(&self) -> Result<usize, LlmError> {
        let mut filters = HashMap::new();
        filters.insert(
            "label".to_string(),
            vec![
                "percy.managed=true".to_string(),
                format!("percy.lens={}", self.config.lens),
                "percy.provider=claude_container".to_string(),
            ],
        );
        filters.insert("status".to_string(), vec!["running".to_string()]);

        let options = ListContainersOptions {
            all: false,
            filters,
            ..Default::default()
        };

        let containers = self
            .docker
            .list_containers(Some(options))
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to list containers: {}", e),
            })?;

        let mut recovered = 0;
        for container in &containers {
            let labels = match &container.labels {
                Some(l) => l,
                None => continue,
            };

            let thread_id_str = match labels.get("percy.thread_id") {
                Some(s) => s,
                None => continue,
            };

            let thread_id = match Uuid::parse_str(thread_id_str) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let container_id = match &container.id {
                Some(id) => id.clone(),
                None => continue,
            };

            // Skip if we already have this session.
            if self.sessions.lock().await.contains_key(&thread_id) {
                continue;
            }

            // Re-attach to the running container.
            let attach_options = AttachContainerOptions::<String> {
                stdin: Some(true),
                stdout: Some(true),
                stderr: Some(true),
                stream: Some(true),
                ..Default::default()
            };

            match self
                .docker
                .attach_container(&container_id, Some(attach_options))
                .await
            {
                Ok(attach_results) => {
                    let session = ContainerSession {
                        container_id: container_id.clone(),
                        thread_id,
                        stdin: Box::pin(attach_results.input),
                        stdout: BollardLineReader::new(attach_results.output),
                        messages_sent: 0, // unknown — will send full history on next exchange
                        cli_session_id: None,
                    };
                    self.sessions.lock().await.insert(thread_id, session);
                    recovered += 1;

                    let container_name = container
                        .names
                        .as_ref()
                        .and_then(|n| n.first().cloned())
                        .unwrap_or_else(|| container_id.clone());
                    tracing::info!(
                        container = %container_name,
                        thread_id = %thread_id,
                        "Re-attached to existing container"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        container_id = %container_id,
                        error = %e,
                        "Failed to re-attach to container, skipping"
                    );
                }
            }
        }

        if recovered > 0 {
            tracing::info!(
                recovered,
                lens = %self.config.lens,
                "Discovered existing container sessions"
            );
        }

        Ok(recovered)
    }

    /// Shut down all container sessions and remove their containers.
    pub async fn shutdown_all(&self) -> Result<(), LlmError> {
        let thread_ids: Vec<Uuid> = {
            let sessions = self.sessions.lock().await;
            sessions.keys().copied().collect()
        };

        for thread_id in thread_ids {
            if let Err(e) = self.remove_session(thread_id).await {
                tracing::warn!(
                    thread_id = %thread_id,
                    error = %e,
                    "Failed to remove container session during shutdown"
                );
            }
        }

        Ok(())
    }

    /// Get the number of active sessions.
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// Check if a thread has an active session.
    pub async fn has_session(&self, thread_id: &Uuid) -> bool {
        self.sessions.lock().await.contains_key(thread_id)
    }

    /// Get the number of messages sent in a session (for delta tracking).
    pub async fn messages_sent(&self, thread_id: &Uuid) -> Option<usize> {
        self.sessions
            .lock()
            .await
            .get(thread_id)
            .map(|s| s.messages_sent)
    }

    /// Update the messages_sent counter for a session.
    pub async fn set_messages_sent(&self, thread_id: &Uuid, count: usize) {
        if let Some(session) = self.sessions.lock().await.get_mut(thread_id) {
            session.messages_sent = count;
        }
    }
}

/// Wait for Claude's system/init message on stdout.
///
/// Returns the session_id if present.
async fn wait_for_system_init(
    reader: &mut BollardLineReader,
) -> Result<Option<String>, LlmError> {
    use tokio::io::AsyncBufReadExt;

    let mut buf = String::new();
    loop {
        buf.clear();
        let bytes_read = reader
            .read_line(&mut buf)
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to read init message: {}", e),
            })?;

        if bytes_read == 0 {
            return Err(LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: "Container process exited before sending init message".to_string(),
            });
        }

        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<ClaudeStreamMessage>(trimmed) {
            Ok(ClaudeStreamMessage::System { session_id }) => {
                return Ok(session_id);
            }
            Ok(_) => {
                tracing::debug!("Ignoring non-system message during init: {}", trimmed);
                continue;
            }
            Err(_) => {
                tracing::debug!("Ignoring unparseable output during init: {}", trimmed);
                continue;
            }
        }
    }
}

/// Generate a container name from lens and thread_id.
pub fn container_name(lens: &str, thread_id: &Uuid) -> String {
    let short_id = &thread_id.to_string()[..8];
    format!("claude-{}-{}", lens, short_id)
}

/// Generate labels for a Percy-managed container.
pub fn container_labels(lens: &str, thread_id: &Uuid) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    labels.insert("percy.managed".into(), "true".into());
    labels.insert("percy.lens".into(), lens.into());
    labels.insert("percy.thread_id".into(), thread_id.to_string());
    labels.insert("percy.provider".into(), "claude_container".into());
    labels
}

#[async_trait::async_trait]
impl ContainerBackend for ContainerPool {
    async fn get_or_create(&self, thread_id: Uuid) -> Result<(), LlmError> {
        self.get_or_create(thread_id).await
    }

    async fn exchange(
        &self,
        thread_id: Uuid,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError> {
        self.exchange(thread_id, prompt).await
    }

    async fn remove_session(&self, thread_id: Uuid) -> Result<(), LlmError> {
        self.remove_session(thread_id).await
    }

    async fn shutdown_all(&self) -> Result<(), LlmError> {
        self.shutdown_all().await
    }

    async fn session_count(&self) -> usize {
        self.session_count().await
    }

    async fn messages_sent(&self, thread_id: &Uuid) -> Option<usize> {
        self.messages_sent(thread_id).await
    }

    async fn set_messages_sent(&self, thread_id: &Uuid, count: usize) {
        self.set_messages_sent(thread_id, count).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- container_name tests ---

    #[test]
    fn test_container_name_format() {
        let thread_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let name = container_name("andrew", &thread_id);
        assert_eq!(name, "claude-andrew-550e8400");
    }

    #[test]
    fn test_container_name_different_lens() {
        let thread_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_ne!(
            container_name("andrew", &thread_id),
            container_name("grace", &thread_id)
        );
    }

    #[test]
    fn test_container_name_different_threads() {
        let t1 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let t2 = Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_ne!(container_name("andrew", &t1), container_name("andrew", &t2));
    }

    // --- container_labels tests ---

    #[test]
    fn test_container_labels_keys() {
        let thread_id = Uuid::new_v4();
        let labels = container_labels("andrew", &thread_id);
        assert_eq!(labels.get("percy.managed"), Some(&"true".to_string()));
        assert_eq!(labels.get("percy.lens"), Some(&"andrew".to_string()));
        assert_eq!(
            labels.get("percy.thread_id"),
            Some(&thread_id.to_string())
        );
        assert_eq!(
            labels.get("percy.provider"),
            Some(&"claude_container".to_string())
        );
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_container_labels_thread_id_is_full_uuid() {
        let thread_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let labels = container_labels("andrew", &thread_id);
        // Labels use the full UUID (unlike container names which use short).
        assert_eq!(
            labels.get("percy.thread_id"),
            Some(&"550e8400-e29b-41d4-a716-446655440000".to_string())
        );
    }

    // --- ContainerPoolConfig tests ---

    #[test]
    fn test_config_clone() {
        let config = ContainerPoolConfig {
            image: "percy-claude:latest".to_string(),
            lens: "andrew".to_string(),
            model: "sonnet".to_string(),
            network: "percy-net".to_string(),
            auth_volume: "claude-auth".to_string(),
            lens_data_volume: Some("claude-data-andrew".to_string()),
            lens_config_path: Some("/path/to/config".to_string()),
            skip_permissions: true,
            request_timeout_secs: 300,
            extra_env: vec!["FOO=bar".to_string()],
        };
        let cloned = config.clone();
        assert_eq!(cloned.image, "percy-claude:latest");
        assert_eq!(cloned.lens, "andrew");
        assert_eq!(cloned.lens_data_volume, Some("claude-data-andrew".to_string()));
    }

    // --- BollardLineReader tests ---

    #[tokio::test]
    async fn test_bollard_line_reader_stdout_only() {
        use tokio::io::AsyncBufReadExt;

        // Simulate a stream with stdout and stderr items.
        let items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![
            Ok(bollard::container::LogOutput::StdOut {
                message: bytes::Bytes::from(r#"{"type":"system","session_id":"s1"}"#.to_string() + "\n"),
            }),
            Ok(bollard::container::LogOutput::StdErr {
                message: bytes::Bytes::from("some debug log\n"),
            }),
            Ok(bollard::container::LogOutput::StdOut {
                message: bytes::Bytes::from(
                    r#"{"type":"result","result":"ok","input_tokens":1,"output_tokens":1}"#.to_string() + "\n",
                ),
            }),
        ];
        let stream = futures::stream::iter(items);
        let mut reader = BollardLineReader::new(stream);

        // First line should be the system message.
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        assert!(n > 0);
        assert!(line.contains("system"));

        // Second line should be the result (stderr skipped).
        line.clear();
        let n = reader.read_line(&mut line).await.unwrap();
        assert!(n > 0);
        assert!(line.contains("result"));
    }

    #[tokio::test]
    async fn test_bollard_line_reader_eof() {
        use tokio::io::AsyncBufReadExt;

        let items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![];
        let stream = futures::stream::iter(items);
        let mut reader = BollardLineReader::new(stream);

        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn test_wait_for_system_init_success() {
        let items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![
            Ok(bollard::container::LogOutput::StdOut {
                message: bytes::Bytes::from(
                    r#"{"type":"system","subtype":"init","session_id":"sess-abc"}"#.to_string() + "\n",
                ),
            }),
        ];
        let stream = futures::stream::iter(items);
        let mut reader = BollardLineReader::new(stream);

        let result = wait_for_system_init(&mut reader).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("sess-abc".to_string()));
    }

    #[tokio::test]
    async fn test_wait_for_system_init_eof() {
        let items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![];
        let stream = futures::stream::iter(items);
        let mut reader = BollardLineReader::new(stream);

        let result = wait_for_system_init(&mut reader).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_wait_for_system_init_skips_non_system() {
        let items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![
            Ok(bollard::container::LogOutput::StdOut {
                message: bytes::Bytes::from("not json\n"),
            }),
            Ok(bollard::container::LogOutput::StdOut {
                message: bytes::Bytes::from(
                    r#"{"type":"assistant","message":{"role":"assistant","content":[]}}"#.to_string() + "\n",
                ),
            }),
            Ok(bollard::container::LogOutput::StdOut {
                message: bytes::Bytes::from(
                    r#"{"type":"system","session_id":"sess-xyz"}"#.to_string() + "\n",
                ),
            }),
        ];
        let stream = futures::stream::iter(items);
        let mut reader = BollardLineReader::new(stream);

        let result = wait_for_system_init(&mut reader).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("sess-xyz".to_string()));
    }

    // --- New tests: session / naming / config ---

    /// 1. Session lookup returns None for unknown thread.
    #[tokio::test]
    async fn test_session_lookup_returns_none_for_unknown_thread() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let random_id = Uuid::new_v4();
        let locked = sessions.lock().await;
        assert!(locked.get(&random_id).is_none());
    }

    /// 2. Session insert and lookup.
    #[tokio::test]
    async fn test_session_insert_and_lookup() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let thread_id = Uuid::new_v4();

        // Create a minimal ContainerSession using an empty stream.
        let empty_items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> =
            vec![];
        let stream = futures::stream::iter(empty_items);
        let session = ContainerSession {
            container_id: "test-container-1".to_string(),
            thread_id,
            stdin: Box::pin(tokio::io::sink()),
            stdout: BollardLineReader::new(stream),
            messages_sent: 0,
            cli_session_id: None,
        };

        sessions.lock().await.insert(thread_id, session);
        let locked = sessions.lock().await;
        assert!(locked.contains_key(&thread_id));
        assert_eq!(locked.get(&thread_id).unwrap().container_id, "test-container-1");
    }

    /// 3. Session update cli_session_id.
    #[tokio::test]
    async fn test_session_update_cli_session_id() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let thread_id = Uuid::new_v4();

        let empty_items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> =
            vec![];
        let session = ContainerSession {
            container_id: "c1".to_string(),
            thread_id,
            stdin: Box::pin(tokio::io::sink()),
            stdout: BollardLineReader::new(futures::stream::iter(empty_items)),
            messages_sent: 0,
            cli_session_id: None,
        };
        sessions.lock().await.insert(thread_id, session);

        // Update cli_session_id.
        {
            let mut locked = sessions.lock().await;
            let s = locked.get_mut(&thread_id).unwrap();
            assert!(s.cli_session_id.is_none());
            s.cli_session_id = Some("sess-updated".to_string());
        }

        let locked = sessions.lock().await;
        assert_eq!(
            locked.get(&thread_id).unwrap().cli_session_id,
            Some("sess-updated".to_string())
        );
    }

    /// 4. Session update messages_sent.
    #[tokio::test]
    async fn test_session_update_messages_sent() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let thread_id = Uuid::new_v4();

        let empty_items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> =
            vec![];
        let session = ContainerSession {
            container_id: "c1".to_string(),
            thread_id,
            stdin: Box::pin(tokio::io::sink()),
            stdout: BollardLineReader::new(futures::stream::iter(empty_items)),
            messages_sent: 0,
            cli_session_id: None,
        };
        sessions.lock().await.insert(thread_id, session);

        {
            let mut locked = sessions.lock().await;
            let s = locked.get_mut(&thread_id).unwrap();
            s.messages_sent += 1;
            s.messages_sent += 1;
        }

        let locked = sessions.lock().await;
        assert_eq!(locked.get(&thread_id).unwrap().messages_sent, 2);
    }

    /// 5. Session remove clears entry.
    #[tokio::test]
    async fn test_session_remove_clears_entry() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let thread_id = Uuid::new_v4();

        let empty_items: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> =
            vec![];
        let session = ContainerSession {
            container_id: "c1".to_string(),
            thread_id,
            stdin: Box::pin(tokio::io::sink()),
            stdout: BollardLineReader::new(futures::stream::iter(empty_items)),
            messages_sent: 0,
            cli_session_id: None,
        };
        sessions.lock().await.insert(thread_id, session);

        assert!(sessions.lock().await.remove(&thread_id).is_some());
        assert!(!sessions.lock().await.contains_key(&thread_id));
    }

    /// 6. Sessions isolated by thread_id — modifying one does not affect the other.
    #[tokio::test]
    async fn test_sessions_isolated_by_thread_id() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let tid1 = Uuid::new_v4();
        let tid2 = Uuid::new_v4();

        for (tid, cid) in [(tid1, "c1"), (tid2, "c2")] {
            let empty: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![];
            let session = ContainerSession {
                container_id: cid.to_string(),
                thread_id: tid,
                stdin: Box::pin(tokio::io::sink()),
                stdout: BollardLineReader::new(futures::stream::iter(empty)),
                messages_sent: 0,
                cli_session_id: None,
            };
            sessions.lock().await.insert(tid, session);
        }

        // Modify session 1 only.
        {
            let mut locked = sessions.lock().await;
            locked.get_mut(&tid1).unwrap().messages_sent = 42;
        }

        let locked = sessions.lock().await;
        assert_eq!(locked.get(&tid1).unwrap().messages_sent, 42);
        assert_eq!(locked.get(&tid2).unwrap().messages_sent, 0);
    }

    /// 7. Remove session leaves others intact.
    #[tokio::test]
    async fn test_remove_session_leaves_others_intact() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let tid1 = Uuid::new_v4();
        let tid2 = Uuid::new_v4();

        for (tid, cid) in [(tid1, "c1"), (tid2, "c2")] {
            let empty: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![];
            let session = ContainerSession {
                container_id: cid.to_string(),
                thread_id: tid,
                stdin: Box::pin(tokio::io::sink()),
                stdout: BollardLineReader::new(futures::stream::iter(empty)),
                messages_sent: 0,
                cli_session_id: None,
            };
            sessions.lock().await.insert(tid, session);
        }

        sessions.lock().await.remove(&tid1);
        let locked = sessions.lock().await;
        assert!(!locked.contains_key(&tid1));
        assert!(locked.contains_key(&tid2));
        assert_eq!(locked.get(&tid2).unwrap().container_id, "c2");
    }

    /// 8. Multiple sessions stored and all retrievable.
    #[tokio::test]
    async fn test_multiple_sessions_all_retrievable() {
        let sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let mut thread_ids = Vec::new();
        for i in 0..5 {
            let tid = Uuid::new_v4();
            thread_ids.push(tid);
            let empty: Vec<Result<bollard::container::LogOutput, bollard::errors::Error>> = vec![];
            let session = ContainerSession {
                container_id: format!("container-{}", i),
                thread_id: tid,
                stdin: Box::pin(tokio::io::sink()),
                stdout: BollardLineReader::new(futures::stream::iter(empty)),
                messages_sent: i,
                cli_session_id: None,
            };
            sessions.lock().await.insert(tid, session);
        }

        let locked = sessions.lock().await;
        assert_eq!(locked.len(), 5);
        for (i, tid) in thread_ids.iter().enumerate() {
            let s = locked.get(tid).unwrap();
            assert_eq!(s.container_id, format!("container-{}", i));
            assert_eq!(s.messages_sent, i);
        }
    }

    /// 9. Container naming with empty lens.
    #[test]
    fn test_container_name_empty_lens() {
        let thread_id = Uuid::parse_str("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee").unwrap();
        let name = container_name("", &thread_id);
        assert_eq!(name, "claude--aaaaaaaa");
        // Verify the name is still valid (no panic, starts with "claude-").
        assert!(name.starts_with("claude-"));
    }

    /// 10. Container naming with special characters in lens name.
    #[test]
    fn test_container_name_special_chars_in_lens() {
        let thread_id = Uuid::parse_str("11111111-2222-3333-4444-555555555555").unwrap();
        let name = container_name("my-lens_v2", &thread_id);
        assert_eq!(name, "claude-my-lens_v2-11111111");

        // Spaces and dots.
        let name2 = container_name("lens.with spaces", &thread_id);
        assert!(name2.contains("lens.with spaces"));
    }

    /// 11. Container labels contain the full UUID for thread_id.
    #[test]
    fn test_container_labels_thread_id_full_uuid() {
        let thread_id = Uuid::parse_str("12345678-1234-1234-1234-123456789abc").unwrap();
        let labels = container_labels("test-lens", &thread_id);
        assert_eq!(
            labels.get("percy.thread_id"),
            Some(&"12345678-1234-1234-1234-123456789abc".to_string())
        );
        assert_eq!(labels.get("percy.lens"), Some(&"test-lens".to_string()));
    }

    /// 12. Container labels with empty lens.
    #[test]
    fn test_container_labels_empty_lens() {
        let thread_id = Uuid::new_v4();
        let labels = container_labels("", &thread_id);
        assert_eq!(labels.get("percy.lens"), Some(&String::new()));
        assert_eq!(labels.get("percy.managed"), Some(&"true".to_string()));
        assert_eq!(labels.len(), 4);
    }

    /// 13. Config request_timeout_secs field is preserved.
    #[test]
    fn test_config_request_timeout_preserved() {
        let config = ContainerPoolConfig {
            image: "img".to_string(),
            lens: "l".to_string(),
            model: "m".to_string(),
            network: "n".to_string(),
            auth_volume: "a".to_string(),
            lens_data_volume: None,
            lens_config_path: None,
            skip_permissions: false,
            request_timeout_secs: 600,
            extra_env: vec![],
        };
        assert_eq!(config.request_timeout_secs, 600);
    }

    /// 14. Extra env vars are preserved through clone.
    #[test]
    fn test_config_extra_env_preserved() {
        let config = ContainerPoolConfig {
            image: "img".to_string(),
            lens: "andrew".to_string(),
            model: "m".to_string(),
            network: "n".to_string(),
            auth_volume: "a".to_string(),
            lens_data_volume: None,
            lens_config_path: None,
            skip_permissions: false,
            request_timeout_secs: 300,
            extra_env: vec![
                "CALLBACK_HOST=http://host:3001".to_string(),
                "GATEWAY_AUTH_TOKEN=tok-123".to_string(),
            ],
        };
        let cloned = config.clone();
        assert_eq!(cloned.extra_env.len(), 2);
        assert!(cloned.extra_env.iter().any(|e| e.starts_with("CALLBACK_HOST=")));
        assert!(cloned.extra_env.iter().any(|e| e.starts_with("GATEWAY_AUTH_TOKEN=")));
    }

    /// 15. Config optional volumes are None by default when not set.
    #[test]
    fn test_config_optional_volumes_none() {
        let config = ContainerPoolConfig {
            image: "img".to_string(),
            lens: "test".to_string(),
            model: "m".to_string(),
            network: "n".to_string(),
            auth_volume: "a".to_string(),
            lens_data_volume: None,
            lens_config_path: None,
            skip_permissions: false,
            request_timeout_secs: 300,
            extra_env: vec![],
        };
        assert!(config.lens_data_volume.is_none());
        assert!(config.lens_config_path.is_none());
    }
}
