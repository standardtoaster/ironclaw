//! Integration tests for ClaudeContainerProvider.
//!
//! These tests require:
//! - Docker or Podman socket accessible
//! - `percy-claude:latest` image available (or any image with `claude` binary)
//!
//! Run with: cargo test --test claude_container_integration -- --ignored

/// Helper to check if Docker/Podman is available.
async fn docker_available() -> bool {
    match bollard::Docker::connect_with_socket_defaults() {
        Ok(docker) => docker.ping().await.is_ok(),
        Err(_) => false,
    }
}

#[tokio::test]
#[ignore] // requires Docker + percy-claude image
async fn test_container_pool_create_and_cleanup() {
    if !docker_available().await {
        eprintln!("Skipping: Docker/Podman not available");
        return;
    }

    use ironclaw::llm::container_pool::{container_labels, container_name, ContainerPool, ContainerPoolConfig};
    use uuid::Uuid;

    let thread_id = Uuid::new_v4();
    let config = ContainerPoolConfig {
        image: std::env::var("TEST_CLAUDE_IMAGE")
            .unwrap_or_else(|_| "percy-claude:latest".to_string()),
        lens: "test".to_string(),
        model: "sonnet".to_string(),
        network: "".to_string(), // no network for test
        auth_volume: "claude-auth".to_string(),
        lens_data_volume: None,
        lens_config_path: None,
        skip_permissions: true,
        request_timeout_secs: 60,
        extra_env: vec![],
    };

    let socket = std::env::var("DOCKER_HOST")
        .unwrap_or_else(|_| "unix:///var/run/docker.sock".to_string());

    // Test container name and labels
    let name = container_name("test", &thread_id);
    assert!(name.starts_with("claude-test-"));

    let labels = container_labels("test", &thread_id);
    assert_eq!(labels.get("percy.managed"), Some(&"true".to_string()));

    // Test pool creation (just verifies Docker connection)
    let pool_result = ContainerPool::new(&socket, config).await;
    assert!(pool_result.is_ok(), "Failed to create pool: {:?}", pool_result.err());

    let pool = pool_result.expect("pool creation verified above");
    assert_eq!(pool.session_count().await, 0);

    // Cleanup
    pool.shutdown_all().await.ok();
}

#[tokio::test]
#[ignore] // requires Docker + percy-claude image
async fn test_container_pool_discover_existing() {
    if !docker_available().await {
        eprintln!("Skipping: Docker/Podman not available");
        return;
    }

    use ironclaw::llm::container_pool::{ContainerPool, ContainerPoolConfig};

    let config = ContainerPoolConfig {
        image: std::env::var("TEST_CLAUDE_IMAGE")
            .unwrap_or_else(|_| "percy-claude:latest".to_string()),
        lens: "test-discover".to_string(),
        model: "sonnet".to_string(),
        network: "".to_string(),
        auth_volume: "claude-auth".to_string(),
        lens_data_volume: None,
        lens_config_path: None,
        skip_permissions: true,
        request_timeout_secs: 60,
        extra_env: vec![],
    };

    let socket = std::env::var("DOCKER_HOST")
        .unwrap_or_else(|_| "unix:///var/run/docker.sock".to_string());

    let pool = ContainerPool::new(&socket, config).await.expect("pool creation");

    // Discover should succeed even with no containers
    let count = pool.discover_existing().await.expect("discover");
    // Count may be 0 or >0 depending on prior test runs; just verify it doesn't error
    eprintln!("Discovered {} existing containers", count);

    pool.shutdown_all().await.ok();
}

#[tokio::test]
#[ignore] // requires Docker + percy-claude image + Claude auth
async fn test_full_container_lifecycle() {
    // This test creates a real container, sends a message, and tears down.
    // Only run manually when percy-claude:latest is built and auth is configured.
    if !docker_available().await {
        eprintln!("Skipping: Docker/Podman not available");
        return;
    }

    use ironclaw::llm::container_pool::{ContainerPool, ContainerPoolConfig};
    use uuid::Uuid;

    let thread_id = Uuid::new_v4();
    let config = ContainerPoolConfig {
        image: "percy-claude:latest".to_string(),
        lens: "test-lifecycle".to_string(),
        model: "sonnet".to_string(),
        network: "".to_string(),
        auth_volume: "claude-auth".to_string(),
        lens_data_volume: None,
        lens_config_path: None,
        skip_permissions: true,
        request_timeout_secs: 120,
        extra_env: vec![],
    };

    let socket = std::env::var("DOCKER_HOST")
        .unwrap_or_else(|_| "unix:///var/run/docker.sock".to_string());

    let pool = ContainerPool::new(&socket, config).await.expect("pool creation");

    // 1. Create container
    pool.get_or_create(thread_id).await.expect("container creation");
    assert!(pool.has_session(&thread_id).await);
    assert_eq!(pool.session_count().await, 1);

    // 2. Send a simple message
    let result = pool.exchange(thread_id, "Say hello in exactly 3 words.").await;
    assert!(result.is_ok(), "Exchange failed: {:?}", result.err());

    // 3. Verify we got a response
    match result.expect("exchange verified above") {
        ironclaw::llm::claude_protocol::ExchangeResult::Complete { content, .. } => {
            assert!(!content.is_empty(), "Empty response from Claude");
            eprintln!("Claude said: {}", content);
        }
        other => panic!("Expected Complete, got: {:?}", other),
    }

    // 4. Remove session
    pool.remove_session(thread_id).await.expect("remove session");
    assert!(!pool.has_session(&thread_id).await);
    assert_eq!(pool.session_count().await, 0);
}
