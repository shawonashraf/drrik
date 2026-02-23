"""
Test wandb and settings integration.

Tests run with pytest.
"""


def test_settings_imports():
    """Test settings module imports."""


def test_settings_creation():
    """Test that settings can be created."""
    from drrik.settings import EnvironmentSettings, WandbConfig

    env_settings = EnvironmentSettings()
    assert env_settings.wandb_project == "drrik-experiments"
    assert env_settings.wandb_mode == "online"

    wandb_cfg = WandbConfig(
        project="test-project",
        config={"test": True},
        enabled=False,
    )
    assert wandb_cfg.project == "test-project"


def test_environment_settings_properties():
    """Test EnvironmentSettings properties."""
    from drrik.settings import EnvironmentSettings

    settings = EnvironmentSettings()
    # These should work without error
    assert isinstance(settings.use_wandb, bool)
    assert isinstance(settings.has_hf_token, bool)


def test_wandb_config_disabled():
    """Test WandbConfig with wandb disabled."""
    from drrik.settings import WandbConfig

    wandb_config = WandbConfig(
        enabled=False,  # Explicitly disable
    )
    assert wandb_config.enabled is False


def test_wandb_config_properties():
    """Test WandbConfig methods when disabled."""
    from drrik.settings import WandbConfig

    wandb_config = WandbConfig(
        project="test-project",
        config={"test": True},
        enabled=False,
    )

    # These should return None or not crash when disabled
    assert wandb_config.get_run_url() is None
    assert wandb_config.get_run_id() is None

    # Logging methods should not crash
    wandb_config.log_metrics({"test": 1.0})
    wandb_config.log_histogram([1, 2, 3], "test")


def test_get_settings_singleton():
    """Test that get_settings returns a singleton."""
    from drrik.settings import get_settings, EnvironmentSettings

    settings1 = get_settings()
    settings2 = get_settings()
    # Should return the same instance
    assert settings1 is settings2
    assert isinstance(settings1, EnvironmentSettings)


def test_main_package_exports():
    """Test main package exports."""
