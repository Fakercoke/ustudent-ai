import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def isolated_operations_db(tmp_path, monkeypatch):
    """No test may write dashboard events into the developer's real database."""
    monkeypatch.setenv("OPS_DB_PATH", str(tmp_path / "ops.sqlite3"))
    monkeypatch.setenv("OPS_ADMIN_PASSWORD", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
