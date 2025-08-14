import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.database import DatabaseManager


@pytest.fixture(scope="session")
def anyio_backend():
    """
    Required for async tests with httpx.
    See: https://fastapi.tiangolo.com/advanced/async-tests/
    """
    return "asyncio"


@pytest.fixture(scope="module")
async def test_client():
    """
    Creates an async test client for the FastAPI application.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client



@pytest.fixture(scope="function")
def test_db_manager():
    """Provides an in-memory SQLite database manager for isolated test runs."""
    db_manager = DatabaseManager(db_url="sqlite:///:memory:")
    yield db_manager
    db_manager.conn.close()