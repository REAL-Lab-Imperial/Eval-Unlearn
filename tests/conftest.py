"""Pytest configuration for eval-unlearn tests."""
import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )


@pytest.fixture(scope="session", autouse=True)
def _preimport_eval_unlearn_techniques():
    """Pre-import eval_unlearn.techniques so patch.dict(sys.modules) snapshots include it.

    Without this, the ESDWrapper test's patch.dict(sys.modules, {"esd": mock}) clears
    eval_unlearn.techniques (and torch) from sys.modules on exit, causing subsequent tests
    that need to reload wrappers to segfault on torch reimport.
    """
    import eval_unlearn.techniques  # noqa: F401
