"""Pytest configuration for eval-learn tests."""
import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )


@pytest.fixture(scope="session", autouse=True)
def _preimport_eval_learn_techniques():
    """Pre-import eval_learn.techniques so patch.dict(sys.modules) snapshots include it.

    Without this, the ESDWrapper test's patch.dict(sys.modules, {"esd": mock}) clears
    eval_learn.techniques (and torch) from sys.modules on exit, causing subsequent tests
    that need to reload wrappers to segfault on torch reimport.
    """
    import eval_learn.techniques  # noqa: F401
