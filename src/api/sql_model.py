"""
Author: Jared Moore
Date: September, 2024

Utilities to serialize objects into SQL.
"""

from datetime import datetime, timezone, timedelta
import logging
from typing import Any
from typing import Counter as TypeCounter

from pydantic import ValidationError

from sqlmodel import (
    ForeignKey,
    Field,
    SQLModel,
    JSON,
    Column,
    DateTime,
)
from sqlmodel._compat import SQLModelConfig
from sqlalchemy import func

from mindgames.conditions import Roles, Condition
from mindgames.model import GameModel
from mindgames.game import Game

from mindgames.utils import EXPERIMENT_CONDITIONS

SQLITE_FILE_NAME = "database.db"
SQLITE_URL_FMT = "sqlite:///{filename}"
SQLITE_URL = SQLITE_URL_FMT.format(filename=SQLITE_FILE_NAME)

CONNECT_ARGS = {"check_same_thread": False}

logger = logging.getLogger(__name__)


class Model(SQLModel, table=True):
    """A table to store GameModels"""

    id: int | None = Field(default=None, primary_key=True)
    game_model_type: str = Field()
    data: dict[str, Any] = Field(sa_column=Column(JSON))

    # The below so that the class calls the validator
    model_config = SQLModelConfig(validate_assignment=True)

    def model_post_init(self, __context):
        if self.game_model_type not in EXPERIMENT_CONDITIONS:
            raise ValueError("Not a valid game model type")
        assert self.data
        gm = GameModel(**self.data)
        difficulty = gm.non_solution_difficulty()
        if gm.is_solution():
            difficulty = "solution"
        if self.game_model_type != difficulty:
            raise ValueError(f"Passed GameModel is not of type {self.game_model_type}")


class Scenario(SQLModel, table=True):
    """A table to store scenarios"""

    id: str = Field(primary_key=True)
    cover_story: str = Field()
    persuader_role: str | None = Field(default=None)
    target_role: str | None = Field(default=None)
    attributes: list[str] = Field(sa_column=Column(JSON))


class Round(SQLModel, table=True):
    """A table to store all of the rounds -- the games human participants
    are playing or have played"""

    # NB: we only make rounds once that are ready or are already playing
    # TODO: but how to delete rounds if they are not completed?
    id: int | None = Field(default=None, primary_key=True)

    persuader_id: int | None = Field(default=None, foreign_key="participant.id")
    target_id: int | None = Field(default=None, foreign_key="participant.id")

    # Whether the persuader or target is an LLM and its name
    llm_persuader: str | None = Field(default=None)
    llm_target: str | None = Field(default=None)

    # This flag controls whether we are in Exp. 1 / 2 or in Exp. 3
    targets_values: bool = Field(default=False)

    allow_lying: bool = Field(default=False)

    # These two flags control what we show to the persuader; they serve as controls
    # (We want to see the performance increase when we reveal the motivation and belief state)
    reveal_motivation: bool = Field(default=False)

    reveal_belief: bool = Field(default=False)

    ## The below variables change over the round

    awaiting_target_response: bool = Field(default=False)
    awaiting_persuader_response: bool = Field(default=False)

    # These two are used to not repeatedly spin up llm threaads or spam
    # the receiver
    processing_target_response: bool = Field(default=False)
    processing_persuader_response: bool = Field(default=False)

    # When neither of these is none, the round is complete
    persuader_choice: str | None = Field(default=None)
    target_choice: str | None = Field(default=None)

    # In experiments 1 and 2 what the target should ideally choose
    # and gets rewarded for choosing.
    target_perfect_info_choice: str | None = Field(default=None)

    scenario_id: str = Field(foreign_key="scenario.id")
    game_model_id: int = Field(foreign_key="model.id")

    game_data: dict[str, Any] = Field(sa_column=Column(JSON))

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=False),
            server_default=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=False),
            onupdate=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )

    # The below so that the class calls the validator
    model_config = SQLModelConfig(validate_assignment=True)

    def model_post_init(self, __context):
        if not self.persuader_id and not self.llm_persuader:
            raise ValueError("No persuader specified.")
        if not self.persuader_id and not self.target_id:
            raise ValueError("One of the participants must be human")

        # This should not error
        assert self.get_roles()

        self.awaiting_persuader_response = self.persuader_id is not None
        self.awaiting_target_response = not self.awaiting_persuader_response

        return self

    def get_roles(self) -> Roles:
        """Returns a Roles object to represent this game round"""
        return Roles(
            human_target=self.target_id is not None,
            human_persuader=self.persuader_id is not None,
            llm_persuader=self.llm_persuader,
            llm_target=self.llm_target,
        )

    def condition(self) -> Condition | None:
        """Returns the Condition for this Round."""
        # TODO: what to do about partially completed rounds?
        game = Game(**self.game_data)
        assert game.game_over()
        try:
            condition = Condition(
                roles=Roles(
                    llm_persuader=self.llm_persuader,
                    llm_target=self.llm_target,
                    human_persuader=(
                        self.persuader_id if self.persuader_id is not None else False
                    ),
                    human_target=(
                        self.target_id if self.target_id is not None else False
                    ),
                ),
                targets_values=self.targets_values,
                reveal_motivation=self.reveal_motivation,
                reveal_belief=self.reveal_belief,
                allow_lying=self.allow_lying,
                add_hint=game.add_hint,
            )
            return condition
        except ValidationError:
            logger.error("Could not validate round " % self)
            return None


class SentMessage(SQLModel, table=True):
    """A table to store all of the messages sent by the persuader or target in a round"""

    id: int | None = Field(default=None, primary_key=True)

    content: str = Field()

    flagged: bool = Field()

    flagged_response: str | None = Field(default=None)

    is_target: bool = Field()

    round_id: int = Field(foreign_key="round.id")

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=False),
            server_default=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )


class ExternalUser(SQLModel, table=True):
    """A table to store the ids of external users (e.g. from Prolific)
    mapping to our internal IDs."""

    # NB: It is a bit hacky here to use a uuid and store it as a string not as
    # a uuid but much of the code already uses a string. It is possible there is a
    # collision. I think that just means `participant_init` would fail and would have
    # to be called again, which is fine.
    id: int = Field(primary_key=True)
    external_id: str = Field()  # The Mturk or Prolific Id


class Participant(SQLModel, table=True):
    """A table to store information about our participants"""

    id: int = Field(primary_key=True)

    initial_survey_responses: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    final_survey_responses: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # Whether the particpant is always the 'target', always the 'persuader' or 'either'
    # or not yet initialized, None
    role: str | None = Field(default=None)

    # What kind of payoff models to still play, e.g. 'solution', 'can_win'
    game_model_types_remaining: TypeCounter[str] = Field(sa_column=Column(JSON))

    # Whether the participant has waited too long in the waiting room
    # and should be forced to end the experiment early.
    # (This should only be set on human-human conditions.)
    waited_too_long: bool = Field(default=False)

    # This flag controls whether we are in Exp. 1 / 2 or in Exp. 3
    targets_values: bool | None = Field(default=None)

    allow_lying: bool | None = Field(default=None)

    # These two flags control what we show to the persuader; they serve as controls
    # (We want to see the performance increase when we reveal the motivation and belief state)
    reveal_motivation: bool | None = Field(default=None)

    reveal_belief: bool | None = Field(default=None)

    round_condition: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    ## The below fields we update over the course of the experiment

    entered_waiting_room: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=False))
    )

    # Whether the participant has been approved, paid, and paid a bonus (if relevant)
    # None means the value is not set
    # False if we may want to deny some work (unlikely).
    work_approved: bool | None = Field(default=None)

    # Any feedback the participant gives at the end of the session
    feedback: str | None = Field(default=None)

    # NB: we have to set `use_alter` so that the SQL database knows which tables to create when
    current_round: int | None = Field(
        default=None,
        sa_column=Column(
            ForeignKey(
                "round.id",
                use_alter=True,
            )
        ),
    )

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=False),
            server_default=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=False),
            onupdate=func.now(),  # pylint: disable=not-callable
            nullable=True,
        ),
    )

    # The below so that the class calls the validator
    model_config = SQLModelConfig(validate_assignment=True)

    # NB: we need this post init method for the default factories to be called
    def model_post_init(self, __context):
        if self.role and self.role not in set(["persuader", "target", "either"]):
            raise ValueError("Invalid persuader role")
        if self.game_model_types_remaining:
            remaining = set(
                self.game_model_types_remaining.keys()  # pylint: disable=no-member
            )
            if len(remaining - EXPERIMENT_CONDITIONS) > 0:
                raise ValueError("Invalid conditon.")

    def conditions_assigned(self):
        """Returns whether or not this particpant has been assigned to conditions yet"""
        return (
            self.role is not None
            and self.allow_lying is not None
            and self.targets_values is not None
            and self.reveal_belief is not None
            and self.reveal_motivation is not None
        )

    def waiting_time(self) -> timedelta | None:
        """Returns the time the participant has been waiting in the lobby"""
        if not self.entered_waiting_room:
            return None
        return datetime.now(timezone.utc) - self.entered_waiting_room.replace(
            tzinfo=timezone.utc  # pylint: disable=no-member
        )
