import pytest

from sqlmodel import create_engine, Session, SQLModel

from sqlmodel.pool import StaticPool


from api.sql_model import (
    SQLITE_URL_FMT,
    CONNECT_ARGS,
)

SQLITE_TEST_URL = SQLITE_URL_FMT.format(filename=":memory:")


@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine(
        SQLITE_TEST_URL,
        echo=False,
        connect_args=CONNECT_ARGS,
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="session")
def session_fixture(engine):
    with Session(engine) as session:
        yield session
