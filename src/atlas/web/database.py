"""SQLAlchemy engine and request-scoped session helpers."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from starlette.requests import Request


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> Engine:
    options = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        options.update(pool_size=5, max_overflow=10, pool_timeout=30)
    return create_engine(database_url, **options)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db_session(request: Request) -> Iterator[Session]:
    factory = request.app.state.session_factory
    with factory() as session:
        yield session
