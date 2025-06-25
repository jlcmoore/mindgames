from collections import Counter
from datetime import timedelta

import pytest

from fastapi.testclient import TestClient

from sqlmodel import Session

##

from mindgames.conditions import RATIONAL_TARGET_ROLE, PAIRED_HUMAN_ROLE, Condition

from api.utils import ServerSettings

from api.api import (
    app,
    get_session,
    get_settings,
)

from api.sql_model import (
    Participant,
)

from .context import session_fixture, engine_fixture


@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    def get_settings_override():
        return ServerSettings(
            round_conditions=Counter({"solution": 1, "can-win": 1}),
            condition_num_rounds=Counter(
                {
                    Condition(roles=RATIONAL_TARGET_ROLE): 1,
                    Condition(roles=PAIRED_HUMAN_ROLE): 2,
                }
            ),
            turn_limit=2,
            waiting_room_timeout=timedelta(seconds=2),
            enforce_persuader_target_roles=True,
        )

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_settings] = get_settings_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(name="client2")
def client_fixture_2(session: Session):
    def get_session_override():
        return session

    def get_settings_override():
        return ServerSettings(
            round_conditions=Counter({"solution": 1}),
            enforce_persuader_target_roles=False,
        )

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_settings] = get_settings_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_read_main(client: TestClient):
    response = client.get("/")
    assert response.status_code == 404
    # TODO: later define test cases for Cameron's front end


def test_participant_init(session: Session, client: TestClient):

    response = client.post("/participant_init/", json={"id": "A"})

    assert response.status_code == 200
    par = session.get(Participant, 1)
    assert par.game_model_types_remaining == Counter({"solution": 1, "can-win": 1})

    response = client.post("/participant_init/", json={"id": "B"})

    assert response.status_code == 200


def test_participant_init2(session: Session, client2: TestClient):

    response = client2.post("/participant_init/", json={"id": "A"})

    assert response.status_code == 200
    par = session.get(Participant, 1)
    assert par.game_model_types_remaining == Counter({"solution": 1})


def test_participant_entered_waiting_room(session: Session, client: TestClient):
    response = client.post("/participant_ready/", json={"id": 0})
    assert response.status_code == 400

    # TODO: other tests not cover by non http


def test_current_round(session: Session, client: TestClient):
    response = client.post("/current_round/", json={"participant_id": 0})
    assert response.status_code == 400

    # TODO: other tests not cover by non http


def test_send_message(session: Session, client: TestClient):
    response = client.post(
        "/send_message/",
        json={
            "participant_id": 0,
            "round_id": 0,
            "message_content": "test",
            "thought_content": "",
        },
    )
    assert response.status_code == 400

    # TODO: other tests not cover by non http


def test_retreive_response(session: Session, client: TestClient):
    response = client.post(
        "/retrieve_response/", json={"participant_id": 0, "round_id": 0}
    )
    assert response.status_code == 400

    # TODO: other tests not cover by non http


def test_make_choice(session: Session, client: TestClient):
    response = client.post(
        "/make_choice/", json={"participant_id": 0, "round_id": 0, "choice": "A"}
    )
    assert response.status_code == 400

    # TODO: other tests not cover by non http


def test_round_result(session: Session, client: TestClient):
    response = client.post("/round_result/", json={"participant_id": 0, "round_id": 1})
    assert response.status_code == 400

    # TODO: other tests not cover by non http
