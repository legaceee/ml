"""
Database Engine & Session Initializer for SQLite.
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.database.models import Base

DB_PATH = Path(__file__).resolve().parent.parent.parent / "cyber_ids.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for obtaining DB session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
