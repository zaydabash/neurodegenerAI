"""
Unified Database management using SQLAlchemy and PostgreSQL.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)
Base = declarative_base()

# --- Models ---


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(50), unique=True, index=True)
    age = Column(Float)
    sex = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class NeuroPrediction(Base):
    __tablename__ = "neuro_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    model_type = Column(String(50))  # tabular, mri, eeg
    prediction = Column(Integer)
    probability = Column(Float)
    confidence = Column(Float)
    results_metadata = Column(PickleType)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SocialPost(Base):
    __tablename__ = "social_posts"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    source = Column(String(50))
    url = Column(String(255))
    author = Column(String(100))
    score = Column(Integer, default=0)
    timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrendTopic(Base):
    __tablename__ = "trend_topics"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String(255), nullable=False)
    keywords = Column(PickleType)
    trending_score = Column(Float)
    volume = Column(Integer)
    growth_rate = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


# --- Database Setup ---


class DatabaseManager:
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(settings.db_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """Create all tables in the database."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()


# Global database manager
_db_manager = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db():
    """Dependency for FastAPI sessions."""
    db = get_db_manager().get_session()
    try:
        yield db
    finally:
        db.close()
