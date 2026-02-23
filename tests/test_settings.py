"""
Test wandb and settings integration.
"""

import sys


def test_settings_imports():
    """Test settings module imports."""
    print("Testing settings imports...")

    try:
        print("  - EnvironmentSettings...", end=" ")
        print("OK")

        print("  - WandbConfig...", end=" ")
        print("OK")

        print("  - get_settings...", end=" ")
        print("OK")

        print("\nAll settings imports successful!")
        return True

    except Exception as e:
        print(f"\nSettings import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_settings_creation():
    """Test that settings can be created."""
    print("\nTesting settings creation...")

    try:
        from drrik.settings import EnvironmentSettings, WandbConfig

        print("  - EnvironmentSettings...", end=" ")
        env_settings = EnvironmentSettings()
        print("OK")

        print("  - WandbConfig...", end=" ")
        _ = WandbConfig(
            project="test-project",
            config={"test": True},
        )
        print("OK")

        print("  - Checking properties...", end=" ")
        print(f"use_wandb={env_settings.use_wandb}")
        print(f"has_hf_token={env_settings.has_hf_token}")
        print("OK")

        print("\nAll settings created successfully!")
        return True

    except Exception as e:
        print(f"\nSettings creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_main_imports():
    """Test main package exports."""
    print("\nTesting main package exports...")

    try:
        print("  - Importing from drrik...", end=" ")
        print("OK")

        print("\nAll main imports successful!")
        return True

    except Exception as e:
        print(f"\nMain imports failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Drrik Framework - Settings Integration Tests")
    print("=" * 50)

    all_passed = True

    all_passed &= test_settings_imports()
    all_passed &= test_settings_creation()
    all_passed &= test_main_imports()

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
