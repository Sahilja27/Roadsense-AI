import pytest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.init_db import init_database

# Use in-memory or dedicated test database
TEST_DB_URL = "sqlite:///./test_roadsense.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    init_database(seed_initial_data=True, engine_bind=test_engine, session_maker=TestingSessionLocal)
    yield
    # Cleanup after tests
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_roadsense.db"):
        try:
            os.remove("./test_roadsense.db")
        except Exception:
            pass



@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
