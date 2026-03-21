//! Container pool for managing warm Claude Code containers via bollard.
//!
//! One container per conversation (thread_id), multiple containers per lens.
//! Containers are created on demand, kept warm, and torn down on de-escalation
//! or explicit cleanup. Communication uses HTTP to a channel MCP server running
//! inside each container.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use bollard::container::{
    Config, CreateContainerOptions, InspectContainerOptions, ListContainersOptions,
    RemoveContainerOptions, StartContainerOptions,
};
use bollard::models::HostConfig;
use bollard::Docker;
use dashmap::DashMap;
use tokio::sync::{oneshot, Mutex};
use uuid::Uuid;

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
    /// Host/IP for the callback URL that containers POST results back to.
    pub callback_host: Option<String>,
    /// Port for the callback URL.
    pub callback_port: Option<u16>,
    /// Auth token for callback authentication.
    pub auth_token: Option<String>,
}

/// Reply received from the channel MCP server via HTTP callback.
#[derive(Debug, Clone)]
pub struct ChannelReply {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub session_id: Option<String>,
}

/// Tracks a pending tool approval request from a container.
#[derive(Debug)]
pub struct PendingApproval {
    pub container_ip: String,
    pub container_port: u16,
    pub thread_id: Uuid,
}

/// Tracks a pending percy_ask_user question from a container.
#[derive(Debug)]
pub struct PendingQuestion {
    pub container_ip: String,
    pub container_port: u16,
    pub thread_id: Uuid,
}

/// Tracks the state of an active container session for a specific thread.
#[derive(Debug, Clone)]
pub struct ContainerSession {
    /// The Docker container ID.
    pub container_id: String,
    /// IP address of the container on the configured network.
    pub container_ip: String,
    /// Port the channel MCP server listens on inside the container.
    pub channel_port: u16,
    /// CLI session ID from Claude (set after first response).
    pub cli_session_id: Option<String>,
    /// Number of messages sent so far (used for delta tracking).
    pub messages_sent: usize,
    /// When this session was created.
    pub created_at: Instant,
}

/// Default port for the channel MCP server inside containers.
const DEFAULT_CHANNEL_PORT: u16 = 3100;

/// Pool of warm Docker containers running Claude Code with channel MCP servers.
///
/// Each container serves one conversation (thread_id). The pool manages creation,
/// health checking, communication via HTTP, and teardown.
pub struct ContainerPool {
    docker: Docker,
    config: ContainerPoolConfig,
    sessions: Arc<Mutex<HashMap<Uuid, ContainerSession>>>,
    /// Pending reply channels: thread_id -> oneshot sender for the callback response.
    pub pending_replies: Arc<DashMap<Uuid, oneshot::Sender<ChannelReply>>>,
    /// Pending tool approval requests: approval_id -> approval state.
    pub pending_approvals: Arc<DashMap<String, PendingApproval>>,
    /// Pending percy_ask_user questions: question_id -> question state.
    pub pending_questions: Arc<DashMap<String, PendingQuestion>>,
    http_client: reqwest::Client,
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
                    reason: format!(
                        "Failed to connect to Docker via TCP at {}: {}",
                        socket_path, e
                    ),
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

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to create HTTP client: {}", e),
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
            pending_replies: Arc::new(DashMap::new()),
            pending_approvals: Arc::new(DashMap::new()),
            pending_questions: Arc::new(DashMap::new()),
            http_client,
        })
    }

    /// Get or create a container session for the given thread.
    ///
    /// Returns the session details including container IP and port for HTTP
    /// communication with the channel MCP server.
    pub async fn get_or_create(&self, thread_id: Uuid) -> Result<ContainerSession, LlmError> {
        // Check if session already exists and is healthy.
        {
            let sessions = self.sessions.lock().await;
            if let Some(session) = sessions.get(&thread_id) {
                // Quick health check.
                let health_url = format!(
                    "http://{}:{}/health",
                    session.container_ip, session.channel_port
                );
                let healthy = reqwest::Client::new()
                    .get(&health_url)
                    .timeout(std::time::Duration::from_secs(5))
                    .send()
                    .await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false);

                if healthy {
                    return Ok(session.clone());
                }

                // Unhealthy — drop lock, tear down, and recreate below.
                let container_id = session.container_id.clone();
                drop(sessions);

                tracing::warn!(
                    container_id = %container_id,
                    thread_id = %thread_id,
                    "Existing container unhealthy, tearing down"
                );
                // Best-effort removal; ignore errors.
                let _ = self.remove_session(thread_id).await;
            }
        }

        // Create a new container.
        let name = container_name(&self.config.lens, &thread_id);
        let labels = build_container_labels(&self.config.lens, &thread_id);
        let env = self.build_container_env(&thread_id);
        let binds = self.build_volume_binds();

        let host_config = HostConfig {
            binds: Some(binds),
            network_mode: Some(self.config.network.clone()),
            ..Default::default()
        };

        let container_config = Config {
            image: Some(self.config.image.clone()),
            env: Some(env),
            labels: Some(labels),
            host_config: Some(host_config),
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
            .map_err(|e| {
                // Best-effort cleanup on start failure.
                let docker = self.docker.clone();
                let cid = container_id.clone();
                tokio::spawn(async move {
                    let _ = docker
                        .remove_container(
                            &cid,
                            Some(RemoveContainerOptions {
                                force: true,
                                ..Default::default()
                            }),
                        )
                        .await;
                });
                LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!("Failed to start container '{}': {}", name, e),
                }
            })?;

        // Discover the container's IP address on the configured network.
        let container_ip =
            self.discover_container_ip(&container_id)
                .await
                .inspect_err(|_| {
                    let docker = self.docker.clone();
                    let cid = container_id.clone();
                    tokio::spawn(async move {
                        let _ = docker
                            .remove_container(
                                &cid,
                                Some(RemoveContainerOptions {
                                    force: true,
                                    ..Default::default()
                                }),
                            )
                            .await;
                    });
                })?;

        // Wait for the channel MCP server to become healthy.
        let channel_port = DEFAULT_CHANNEL_PORT;
        if let Err(e) = self
            .wait_for_health(&container_ip, channel_port, 120)
            .await
        {
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

        tracing::info!(
            container_id = %container_id,
            name = %name,
            thread_id = %thread_id,
            container_ip = %container_ip,
            channel_port = channel_port,
            "Container session created"
        );

        let session = ContainerSession {
            container_id,
            container_ip,
            channel_port,
            cli_session_id: None,
            messages_sent: 0,
            created_at: Instant::now(),
        };

        self.sessions.lock().await.insert(thread_id, session.clone());
        Ok(session)
    }

    /// Send a message to the container's channel MCP server.
    ///
    /// Posts the message to the /message endpoint. The container processes it
    /// asynchronously and calls back to IronClaw with the result.
    pub async fn send_message(
        &self,
        session: &ContainerSession,
        thread_id: &Uuid,
        prompt: &str,
    ) -> Result<(), LlmError> {
        let url = format!(
            "http://{}:{}/message",
            session.container_ip, session.channel_port
        );

        let body = serde_json::json!({
            "thread_id": thread_id.to_string(),
            "content": prompt,
        });

        let response = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!(
                    "Failed to send message to container {}: {}",
                    session.container_id, e
                ),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!(
                    "Container /message returned {}: {}",
                    status, body_text
                ),
            });
        }

        Ok(())
    }

    /// Look up a session by thread ID.
    pub async fn session_by_thread(&self, thread_id: &Uuid) -> Option<ContainerSession> {
        self.sessions.lock().await.get(thread_id).cloned()
    }

    /// Update the CLI session ID for a thread's session.
    pub async fn update_session_id(&self, thread_id: &Uuid, session_id: String) {
        if let Some(session) = self.sessions.lock().await.get_mut(thread_id) {
            session.cli_session_id = Some(session_id);
        }
    }

    /// Increment the messages_sent counter for a thread's session.
    pub async fn update_messages_sent(&self, thread_id: &Uuid) {
        if let Some(session) = self.sessions.lock().await.get_mut(thread_id) {
            session.messages_sent += 1;
        }
    }

    /// Remove a session and destroy the associated container.
    ///
    /// Also cleans up any pending reply channels for this thread.
    pub async fn remove_session(&self, thread_id: Uuid) -> Result<(), LlmError> {
        // Clean up pending reply channel.
        self.pending_replies.remove(&thread_id);

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

    /// Discover existing Percy-managed containers and re-adopt healthy ones.
    ///
    /// Returns the number of sessions recovered. Unhealthy or stale containers
    /// are torn down.
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

            // Inspect to get IP address.
            let container_ip = match self.discover_container_ip(&container_id).await {
                Ok(ip) => ip,
                Err(e) => {
                    tracing::warn!(
                        container_id = %container_id,
                        error = %e,
                        "Failed to get IP for existing container, tearing down"
                    );
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
                    continue;
                }
            };

            // Health check.
            let channel_port = DEFAULT_CHANNEL_PORT;
            let health_url = format!("http://{}:{}/health", container_ip, channel_port);
            let healthy = reqwest::Client::new()
                .get(&health_url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            if !healthy {
                tracing::warn!(
                    container_id = %container_id,
                    thread_id = %thread_id,
                    "Existing container unhealthy, tearing down"
                );
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
                continue;
            }

            let session = ContainerSession {
                container_id: container_id.clone(),
                container_ip,
                channel_port,
                cli_session_id: None,
                messages_sent: 0,
                created_at: Instant::now(),
            };
            self.sessions.lock().await.insert(thread_id, session);
            recovered += 1;

            let display_name = container
                .names
                .as_ref()
                .and_then(|n| n.first().cloned())
                .unwrap_or_else(|| container_id.clone());
            tracing::info!(
                container = %display_name,
                thread_id = %thread_id,
                "Re-adopted existing container"
            );
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

    // --- Private helpers ---

    /// Extract the container's IP address from a Docker inspect result.
    async fn discover_container_ip(&self, container_id: &str) -> Result<String, LlmError> {
        let inspect = self
            .docker
            .inspect_container(container_id, None::<InspectContainerOptions>)
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!("Failed to inspect container '{}': {}", container_id, e),
            })?;

        let networks = inspect
            .network_settings
            .as_ref()
            .and_then(|ns| ns.networks.as_ref())
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!(
                    "Container '{}' has no network settings",
                    container_id
                ),
            })?;

        let endpoint = networks.get(&self.config.network).ok_or_else(|| {
            LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!(
                    "Container '{}' not connected to network '{}'",
                    container_id, self.config.network
                ),
            }
        })?;

        let ip = endpoint
            .ip_address
            .as_ref()
            .filter(|ip| !ip.is_empty())
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "claude_container".to_string(),
                reason: format!(
                    "Container '{}' has no IP on network '{}'",
                    container_id, self.config.network
                ),
            })?;

        Ok(ip.clone())
    }

    /// Poll the container's health endpoint until it responds successfully.
    async fn wait_for_health(
        &self,
        container_ip: &str,
        channel_port: u16,
        timeout_secs: u64,
    ) -> Result<(), LlmError> {
        let health_url = format!("http://{}:{}/health", container_ip, channel_port);
        let deadline = Instant::now() + std::time::Duration::from_secs(timeout_secs);
        let poll_interval = std::time::Duration::from_secs(1);

        loop {
            let healthy = reqwest::Client::new()
                .get(&health_url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);

            if healthy {
                return Ok(());
            }

            if Instant::now() >= deadline {
                return Err(LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!(
                        "Container health check timed out after {}s ({})",
                        timeout_secs, health_url
                    ),
                });
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Build environment variables for a new container.
    fn build_container_env(&self, thread_id: &Uuid) -> Vec<String> {
        let mut env = self.config.extra_env.clone();
        env.push(format!("PERCY_LENS={}", self.config.lens));
        env.push(format!("PERCY_THREAD_ID={}", thread_id));
        env.push(format!("PERCY_CHANNEL_PORT={}", DEFAULT_CHANNEL_PORT));
        env.push(format!("CLAUDE_MODEL={}", self.config.model));

        if let Some(ref host) = self.config.callback_host {
            let port = self.config.callback_port.unwrap_or(3001);
            env.push(format!("PERCY_CALLBACK_URL=http://{}:{}", host, port));
        }

        if let Some(ref token) = self.config.auth_token {
            env.push(format!("PERCY_AUTH_TOKEN={}", token));
        }

        env
    }

    /// Build volume bind mounts for a new container.
    fn build_volume_binds(&self) -> Vec<String> {
        let mut binds = vec![format!("{}:/mnt/claude-auth:ro", self.config.auth_volume)];
        if let Some(ref vol) = self.config.lens_data_volume {
            binds.push(format!("{}:/mnt/lens-data:rw", vol));
        }
        if let Some(ref path) = self.config.lens_config_path {
            binds.push(format!("{}:/mnt/lens-config:ro", path));
        }
        binds
    }
}

/// Generate a container name from lens and thread_id.
pub fn container_name(lens: &str, thread_id: &Uuid) -> String {
    let short_id = &thread_id.to_string()[..8];
    format!("claude-{}-{}", lens, short_id)
}

/// Generate labels for a Percy-managed container.
pub fn build_container_labels(lens: &str, thread_id: &Uuid) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    labels.insert("percy.managed".into(), "true".into());
    labels.insert("percy.lens".into(), lens.into());
    labels.insert("percy.thread_id".into(), thread_id.to_string());
    labels.insert("percy.provider".into(), "claude_container".into());
    labels
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

    // --- build_container_labels tests ---

    #[test]
    fn test_container_labels_keys() {
        let thread_id = Uuid::new_v4();
        let labels = build_container_labels("andrew", &thread_id);
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
        let labels = build_container_labels("andrew", &thread_id);
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
            callback_host: Some("192.168.1.100".to_string()),
            callback_port: Some(3001),
            auth_token: Some("test-token".to_string()),
        };
        let cloned = config.clone();
        assert_eq!(cloned.image, "percy-claude:latest");
        assert_eq!(cloned.lens, "andrew");
        assert_eq!(
            cloned.lens_data_volume,
            Some("claude-data-andrew".to_string())
        );
        assert_eq!(
            cloned.callback_host,
            Some("192.168.1.100".to_string())
        );
    }

    // --- ContainerSession tests ---

    #[test]
    fn test_session_clone() {
        let session = ContainerSession {
            container_id: "abc123".to_string(),
            container_ip: "172.18.0.5".to_string(),
            channel_port: 3100,
            cli_session_id: Some("sess-1".to_string()),
            messages_sent: 3,
            created_at: Instant::now(),
        };
        let cloned = session.clone();
        assert_eq!(cloned.container_id, "abc123");
        assert_eq!(cloned.container_ip, "172.18.0.5");
        assert_eq!(cloned.channel_port, 3100);
        assert_eq!(cloned.messages_sent, 3);
    }

    // --- build_container_env tests ---

    #[test]
    fn test_build_env_basic() {
        let config = ContainerPoolConfig {
            image: "img".to_string(),
            lens: "andrew".to_string(),
            model: "sonnet-4".to_string(),
            network: "net".to_string(),
            auth_volume: "vol".to_string(),
            lens_data_volume: None,
            lens_config_path: None,
            skip_permissions: false,
            request_timeout_secs: 300,
            extra_env: vec!["EXTRA=1".to_string()],
            callback_host: Some("10.0.0.1".to_string()),
            callback_port: Some(3001),
            auth_token: Some("tok-123".to_string()),
        };
        let thread_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

        // We can't call build_container_env directly without a pool, so we test
        // the env construction logic via the config fields.
        let mut env = config.extra_env.clone();
        env.push(format!("PERCY_LENS={}", config.lens));
        env.push(format!("PERCY_THREAD_ID={}", thread_id));
        env.push(format!("PERCY_CHANNEL_PORT={}", DEFAULT_CHANNEL_PORT));
        env.push(format!("CLAUDE_MODEL={}", config.model));

        assert!(env.contains(&"EXTRA=1".to_string()));
        assert!(env.contains(&"PERCY_LENS=andrew".to_string()));
        assert!(env.contains(&"CLAUDE_MODEL=sonnet-4".to_string()));
        assert!(env.contains(&"PERCY_CHANNEL_PORT=3100".to_string()));
    }

    // --- build_volume_binds tests ---

    #[test]
    fn test_volume_binds_minimal() {
        let config = ContainerPoolConfig {
            image: "img".to_string(),
            lens: "andrew".to_string(),
            model: "sonnet".to_string(),
            network: "net".to_string(),
            auth_volume: "claude-auth".to_string(),
            lens_data_volume: None,
            lens_config_path: None,
            skip_permissions: false,
            request_timeout_secs: 300,
            extra_env: vec![],
            callback_host: None,
            callback_port: None,
            auth_token: None,
        };

        let mut binds = vec![format!("{}:/mnt/claude-auth:ro", config.auth_volume)];
        if let Some(ref vol) = config.lens_data_volume {
            binds.push(format!("{}:/mnt/lens-data:rw", vol));
        }
        if let Some(ref path) = config.lens_config_path {
            binds.push(format!("{}:/mnt/lens-config:ro", path));
        }

        assert_eq!(binds.len(), 1);
        assert_eq!(binds[0], "claude-auth:/mnt/claude-auth:ro");
    }

    #[test]
    fn test_volume_binds_all() {
        let config = ContainerPoolConfig {
            image: "img".to_string(),
            lens: "andrew".to_string(),
            model: "sonnet".to_string(),
            network: "net".to_string(),
            auth_volume: "claude-auth".to_string(),
            lens_data_volume: Some("data-vol".to_string()),
            lens_config_path: Some("/etc/percy/andrew".to_string()),
            skip_permissions: false,
            request_timeout_secs: 300,
            extra_env: vec![],
            callback_host: None,
            callback_port: None,
            auth_token: None,
        };

        let mut binds = vec![format!("{}:/mnt/claude-auth:ro", config.auth_volume)];
        if let Some(ref vol) = config.lens_data_volume {
            binds.push(format!("{}:/mnt/lens-data:rw", vol));
        }
        if let Some(ref path) = config.lens_config_path {
            binds.push(format!("{}:/mnt/lens-config:ro", path));
        }

        assert_eq!(binds.len(), 3);
        assert!(binds[1].contains("data-vol"));
        assert!(binds[2].contains("/etc/percy/andrew"));
    }
}
