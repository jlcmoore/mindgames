"""
Author: Jared Moore
Date: September, 2024

Contains utility functions and constants for the api.
"""

from collections import Counter
from datetime import timedelta
from typing import Counter as TypeCounter
from typing import Type, Tuple, Any

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
)

from mindgames.utils import EXPERIMENT_CONDITIONS

from mindgames.conditions import (
    Condition,
    Roles,
    RATIONAL_TARGET_ROLE,
    PAIRED_HUMAN_ROLE,
)

DEFAULT_WAITING_ROOM_TIMEOUT = timedelta(seconds=60)

MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER = 5


class ServerSettings(BaseSettings):
    """
    Variables potentially to change when running differently-styled experiments.

    TODO: Currently assuming that we will spin up a different server instance for
    each condition being run
    """

    dev_environment: bool = True

    # TODO: currently assuming participants have to be paired
    # (regardless of playing as persuader or target)
    # in the game of the right type ['solution', 'can_win', 'never-win', 'always-win']
    # We possibly want to make these part of `Condition`
    # TODO: currently assuming participants can play these in any order,
    round_conditions: TypeCounter[str] = Counter(
        {"solution": 3, "can-win": 1, "never-win": 1, "always-win": 1}
    )

    inline_lists: bool = True
    proposals_as_html: bool = True

    turn_limit: int = 5  # > 0

    # Whether participants should always play the target or persuader.
    # If false they can play either.
    enforce_persuader_target_roles: bool = True

    # If True, players can only play one kind of `Condition`
    enforce_player_round_condition: bool = True

    # If there are no participants in the waiting room and there are only paired experiments
    # left to run, whether to 'overstuff' the non-paired conditions
    overassign_non_paired_conditions: bool = True

    # How long should a participant wait until timing out of a round and
    # starting a new one.
    participant_conversation_timeout: timedelta = timedelta(minutes=3)

    # NB: The total number of participants is `condition_num_rounds.total()`
    # possibly with a few extra, as we overassign at times
    # The total number of rounds should be about
    # `condition_num_rounds.total() * round_conditions.total()`
    condition_num_rounds: TypeCounter[Condition] = Counter(
        {
            # Rational target condition -- 10 human persuaders desired
            Condition(
                roles=RATIONAL_TARGET_ROLE, reveal_motivation=False, reveal_belief=False
            ): 20,
            # Human - human condition -- 10 persuaders, 10 targets; n = 20
            # Condition(roles=PAIRED_HUMAN_ROLE): 10,
            # Human - llm condition -- 10 persuaders, 10 instances of gpt-4o as a target; n = 10
            # Coundition(roles=Roles(human_persuader=True, llm_target='gpt-4o')) : 10,
        }
    )

    # NB: This is used to initialize `condition_num_rounds` from a file
    conditions: list[dict[str, Any]] | None = None

    # Whether to add a hint to the high level instructions
    add_hint: bool = False

    # The completion code for the Prolific study
    completion_code: str = "TEST"

    waiting_room_timeout: timedelta = DEFAULT_WAITING_ROOM_TIMEOUT

    # If true allows participants to play each other repeatedly. If false, they always play a
    # new participant
    participants_rematch: bool = False

    # Whether to run any server tasks in the background
    background_tasks: bool = True

    # Whether to allow participants to log their notes in a CoT style
    chain_of_thought: bool = False

    # Whether to use the non-mental scenarios
    non_mental: bool = False

    # When running the server, loads in the settings from the file
    # TODO: verify that the path is correct for this file.
    model_config = SettingsConfigDict(yaml_file="server_settings.yml")

    def model_post_init(self, __context):
        # Initialize the condition counter
        if self.conditions:
            self.condition_num_rounds = Counter()
            for condition in self.conditions:
                roles = Roles(
                    **condition["roles"],
                )
                self.condition_num_rounds[
                    Condition(roles=roles, **condition["condition"])
                ] += condition["count"]
        invalid_conditions = set(self.round_conditions.keys()) - EXPERIMENT_CONDITIONS
        if len(invalid_conditions) > 0:
            raise ValueError(f"Conditions, {invalid_conditions}, are invalid.")
        if self.turn_limit <= 0:
            raise ValueError("Must pass a positive turn limit.")

    # NB: We redefine this method so we can use a yaml settings file
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Allow programmatic initialization
            YamlConfigSettingsSource(settings_cls),  # Allow YAML configuration
        )
