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

    use ironclaw::llm::container_pool::{
        build_container_labels, container_name, ContainerPool, ContainerPoolConfig,
    };
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
        callback_host: None,
        callback_port: None,
        auth_token: None,
    };

    let socket = std::env::var("DOCKER_HOST")
        .unwrap_or_else(|_| "unix:///var/run/docker.sock".to_string());

    // Test container name and labels
    let name = container_name("test", &thread_id);
    assert!(name.starts_with("claude-test-"));

    let labels = build_container_labels("test", &thread_id);
    assert_eq!(labels.get("percy.managed"), Some(&"true".to_string()));

    // Test pool creation (just verifies Docker connection)
    let pool_result = ContainerPool::new(&socket, config).await;
    assert!(
        pool_result.is_ok(),
        "Failed to create pool: {:?}",
        pool_result.err()
    );

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
        callback_host: None,
        callback_port: None,
        auth_token: None,
    };

    let socket = std::env::var("DOCKER_HOST")
        .unwrap_or_else(|_| "unix:///var/run/docker.sock".to_string());

    let pool = ContainerPool::new(&socket, config)
        .await
        .expect("pool creation");

    // Discover should succeed even with no containers
    let count = pool.discover_existing().await.expect("discover");
    // Count may be 0 or >0 depending on prior test runs; just verify it doesn't error
    eprintln!("Discovered {} existing containers", count);

    pool.shutdown_all().await.ok();
}

#[tokio::test]
#[ignore] // requires Docker + percy-claude image + Claude auth + channel MCP
async fn test_full_container_lifecycle() {
    // This test creates a real container, verifies session management, and tears down.
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
        callback_host: None,
        callback_port: None,
        auth_token: None,
    };

    let socket = std::env::var("DOCKER_HOST")
        .unwrap_or_else(|_| "unix:///var/run/docker.sock".to_string());

    let pool = ContainerPool::new(&socket, config)
        .await
        .expect("pool creation");

    // 1. Create container
    let session = pool
        .get_or_create(thread_id)
        .await
        .expect("container creation");
    assert!(pool.has_session(&thread_id).await);
    assert_eq!(pool.session_count().await, 1);
    assert!(!session.container_ip.is_empty());

    // 2. Send a message (async — reply comes via callback, won't arrive in this test)
    let send_result = pool
        .send_message(&session, &thread_id, "Say hello in exactly 3 words.")
        .await;
    // This may fail if the channel MCP server isn't running in the image;
    // that's OK for a lifecycle test — we're testing session management.
    if let Err(e) = &send_result {
        eprintln!("send_message failed (expected if no channel MCP): {}", e);
    }

    // 3. Remove session
    pool.remove_session(thread_id)
        .await
        .expect("remove session");
    assert!(!pool.has_session(&thread_id).await);
    assert_eq!(pool.session_count().await, 0);
}
