"""
Pytest configuration for Drrik tests.
"""


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU (slow)")
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow (skip by default)")
