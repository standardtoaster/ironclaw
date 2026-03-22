//! Activation script runner for skills.
//!
//! Executes a script at skill activation time, captures stdout, and returns it
//! for injection into the skill's prompt context. Failures are logged but never
//! prevent the skill from activating — the skill simply runs without the dynamic
//! context.

use std::path::Path;
use std::time::Duration;

use tokio::process::Command;

use crate::skills::ActivationScript;

/// Safe environment variables forwarded to script subprocesses.
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP",
];

/// Run an activation script and return its stdout, or `None` on any failure.
///
/// The script receives `IRONCLAW_PORT`, `IRONCLAW_TOKEN`, and `IRONCLAW_USER_ID`
/// in its environment. Stdout is truncated to `max_output_bytes`. If the script
/// fails, times out, or produces no output, a warning is logged and `None` is
/// returned so the skill still activates with static content only.
pub async fn run_activation_script(
    script: &ActivationScript,
    skill_dir: &Path,
    port: u16,
    token: &str,
    user_id: &str,
) -> Option<String> {
    // Resolve interpreter binary
    let interpreter = match script.language.as_str() {
        "python" => "python3",
        "bash" => "bash",
        "node" => "node",
        other => {
            tracing::warn!(
                language = other,
                "Unknown activation script language, skipping"
            );
            return None;
        }
    };

    // Build the script path
    let script_path = resolve_script_path(script, skill_dir)?;

    let mut command = Command::new(interpreter);
    command.arg(&script_path);

    // Scrub environment: only safe vars + IRONCLAW_* context
    command.env_clear();
    for var in SAFE_ENV_VARS {
        if let Ok(val) = std::env::var(var) {
            command.env(var, val);
        }
    }
    command.env("IRONCLAW_PORT", port.to_string());
    command.env("IRONCLAW_TOKEN", token);
    command.env("IRONCLAW_USER_ID", user_id);

    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            tracing::warn!(
                interpreter = interpreter,
                error = %e,
                "Failed to spawn activation script"
            );
            return None;
        }
    };

    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();
    let timeout = Duration::from_millis(script.timeout_ms);
    let max_bytes = script.max_output_bytes;

    let result = tokio::time::timeout(timeout, async {
        // Drain stdout and stderr concurrently
        let stdout_fut = async {
            if let Some(mut out) = stdout_handle {
                let mut buf = Vec::with_capacity(max_bytes.min(8192));
                tokio::io::AsyncReadExt::read_buf(&mut out, &mut buf).await.ok();
                // Keep reading until EOF or we have enough
                while buf.len() < max_bytes {
                    let mut chunk = vec![0u8; 4096];
                    match tokio::io::AsyncReadExt::read(&mut out, &mut chunk).await {
                        Ok(0) => break, // EOF
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                        Err(_) => break,
                    }
                }
                buf.truncate(max_bytes);
                buf
            } else {
                Vec::new()
            }
        };

        let stderr_fut = async {
            if let Some(mut err) = stderr_handle {
                let mut buf = Vec::with_capacity(1024);
                // Read a limited amount of stderr for diagnostics
                let _ = tokio::io::AsyncReadExt::read_buf(&mut err, &mut buf).await;
                buf.truncate(2048);
                buf
            } else {
                Vec::new()
            }
        };

        let (stdout_bytes, stderr_bytes) = tokio::join!(stdout_fut, stderr_fut);
        let status = child.wait().await;
        (stdout_bytes, stderr_bytes, status)
    })
    .await;

    match result {
        Ok((stdout_bytes, stderr_bytes, Ok(status))) => {
            if !status.success() {
                let stderr_str = String::from_utf8_lossy(&stderr_bytes);
                tracing::warn!(
                    exit_code = status.code(),
                    stderr = %stderr_str.chars().take(500).collect::<String>(),
                    "Activation script exited with non-zero status"
                );
                return None;
            }
            let output = String::from_utf8_lossy(&stdout_bytes).trim().to_string();
            if output.is_empty() {
                tracing::debug!("Activation script produced no output");
                return None;
            }
            Some(output)
        }
        Ok((_, _, Err(e))) => {
            tracing::warn!(error = %e, "Failed to wait on activation script process");
            None
        }
        Err(_) => {
            tracing::warn!(
                timeout_ms = script.timeout_ms,
                "Activation script timed out, killing process"
            );
            // Best-effort kill
            let _ = child.kill().await;
            None
        }
    }
}

/// Resolve the script to a filesystem path. For inline `source`, writes to a
/// temp file. For `source_file`, resolves relative to the skill directory.
fn resolve_script_path(script: &ActivationScript, skill_dir: &Path) -> Option<String> {
    if let Some(ref source) = script.source {
        // Write inline script to a temp file
        let ext = match script.language.as_str() {
            "python" => "py",
            "bash" => "sh",
            "node" => "js",
            _ => "tmp",
        };
        match write_temp_script(source, ext) {
            Ok(path) => Some(path),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to write inline activation script to temp file");
                None
            }
        }
    } else if let Some(ref source_file) = script.source_file {
        let path = skill_dir.join(source_file);
        if path.exists() {
            path.to_str().map(|s| s.to_string()).or_else(|| {
                tracing::warn!("Activation script path is not valid UTF-8");
                None
            })
        } else {
            tracing::warn!(
                path = %path.display(),
                "Activation script source_file does not exist"
            );
            None
        }
    } else {
        tracing::warn!("Activation script has neither `source` nor `source_file`");
        None
    }
}

/// Write script content to a temporary file and return its path.
fn write_temp_script(content: &str, extension: &str) -> Result<String, std::io::Error> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let dir = std::env::temp_dir();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let filename = format!(
        "ironclaw-skill-{}-{}.{}",
        std::process::id(),
        seq,
        extension
    );
    let path = dir.join(filename);
    std::fs::write(&path, content)?;
    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
    }
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "path not UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_script_prints_hello() {
        let script = ActivationScript {
            language: "bash".to_string(),
            source: Some("echo hello".to_string()),
            source_file: None,
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, Path::new("/tmp"), 3003, "test-token", "test-user")
                .await;
        assert_eq!(result, Some("hello".to_string()));
    }

    #[tokio::test]
    async fn test_script_nonzero_exit_returns_none() {
        let script = ActivationScript {
            language: "bash".to_string(),
            source: Some("exit 1".to_string()),
            source_file: None,
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, Path::new("/tmp"), 3003, "test-token", "test-user")
                .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_script_timeout_returns_none() {
        let script = ActivationScript {
            language: "bash".to_string(),
            source: Some("sleep 10".to_string()),
            source_file: None,
            timeout_ms: 100, // very short timeout
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, Path::new("/tmp"), 3003, "test-token", "test-user")
                .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_script_env_vars_available() {
        let script = ActivationScript {
            language: "bash".to_string(),
            source: Some(
                "echo \"port=$IRONCLAW_PORT token=$IRONCLAW_TOKEN user=$IRONCLAW_USER_ID\""
                    .to_string(),
            ),
            source_file: None,
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, Path::new("/tmp"), 3003, "test-token", "test-user")
                .await;
        assert_eq!(
            result,
            Some("port=3003 token=test-token user=test-user".to_string())
        );
    }

    #[tokio::test]
    async fn test_unknown_language_returns_none() {
        let script = ActivationScript {
            language: "ruby".to_string(),
            source: Some("puts 'hello'".to_string()),
            source_file: None,
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, Path::new("/tmp"), 3003, "test-token", "test-user")
                .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_no_source_or_file_returns_none() {
        let script = ActivationScript {
            language: "bash".to_string(),
            source: None,
            source_file: None,
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, Path::new("/tmp"), 3003, "test-token", "test-user")
                .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_source_file_runs_correctly() {
        // Create a temporary script file
        let dir = tempfile::tempdir().expect("create tempdir");
        let script_path = dir.path().join("test.sh");
        std::fs::write(&script_path, "#!/bin/bash\necho from-file").expect("write script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }

        let script = ActivationScript {
            language: "bash".to_string(),
            source: None,
            source_file: Some("test.sh".to_string()),
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        let result =
            run_activation_script(&script, dir.path(), 3003, "test-token", "test-user").await;
        assert_eq!(result, Some("from-file".to_string()));
    }
}
