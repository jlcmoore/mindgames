"""
Author: Jared Moore
Date: September, 2024

Utilities to query the SQL database.
"""

from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import logging
import random
import statistics
from typing import Iterable, Tuple, Any
from typing import Counter as TypeCounter
import os
import pprint

import numpy as np
import pandas as pd

from sqlmodel import Session, select, or_, and_, asc, desc
from sqlalchemy import func

from mindgames.conditions import Roles, Condition
from mindgames.game import Game
from mindgames.utils import (
    get_payoff_file_path,
    SCENARIOS_FILE,
    NON_MENTAL_SCENARIOS_FILE,
)

from experiments.utils import (
    output_conditions_and_games,
    RESULTS_DIR,
    condition_to_dir,
)


from .sql_model import (
    Model,
    Scenario,
    Round,
    Participant,
    SentMessage,
    ExternalUser,
)

from .utils import ServerSettings

logger = logging.getLogger(__name__)


def min_positive_timedelta_diff(td1: timedelta, td2: timedelta) -> timedelta:
    """
    Computes the differences between the two time deltas and returns the minimum
    """
    diff1 = abs(td1 - td2)
    diff2 = abs(td2 - td1)

    min_diff = min(diff1, diff2)

    return min_diff


def exclude_round_messages(session: Session, rd: Round) -> (bool, bool):
    """
    Counts the time each message took to send and its characters in Round.
    Returns a (bool, bool) if the messages where too_short or too_quick
    (fewer than 10 characters, shorter than 5 seconds). Only applies
    to humans.
    """
    messages = session.exec(
        select(SentMessage)
        .where(SentMessage.round_id == rd.id)
        .where(SentMessage.flagged.is_not(True))  # pylint: disable=no-member
        .order_by(asc(SentMessage.created_at), desc(SentMessage.is_target))
        # earliest to latest, then whether it is the target or not
    ).all()

    target_lengths = []
    persuader_lengths = []
    target_times = []
    persuader_times = []

    # The start of the round plus 30 seconds (how long they have to wait on the
    # instructions page for)
    last_sent = rd.created_at + timedelta(seconds=30)
    last_was_target = None
    for message in messages:
        characters = len(message.content)
        # If somehow we mess up the order
        elapsed_time = min_positive_timedelta_diff(message.created_at, last_sent)
        last_sent = message.created_at
        count_timestamp = not last_was_target or last_was_target != message.is_target
        if not count_timestamp:
            logger.warning("Message out of order. Ignoring time stamp diffs.")
        last_was_target = message.is_target

        # Only count the messages and times for human targets and persuaders
        if message.is_target:
            target_lengths.append(characters)
            if count_timestamp:
                target_times.append(elapsed_time)
        else:
            persuader_lengths.append(characters)
            if count_timestamp:
                persuader_times.append(elapsed_time)

    if rd.persuader_id is not None and rd.target_id is None:
        # We need to swap the times here.
        # Persuader messages and LLM response get added at the same time.
        persuader_times, target_times = target_times, persuader_times

    too_short = False
    too_quick = False

    if persuader_lengths and rd.persuader_id is not None:
        assert persuader_times
        avg_persuader_length = np.mean(persuader_lengths)
        avg_persuader_time = np.mean(persuader_times)
        too_short |= avg_persuader_length < 10
        too_quick |= avg_persuader_time < timedelta(seconds=5)

    if target_lengths and rd.target_id is not None:
        assert target_times
        avg_target_length = np.mean(target_lengths)
        avg_target_time = np.mean(target_times)

        too_short |= avg_target_length < 10
        too_quick |= avg_target_time < timedelta(seconds=5)

    if too_short:
        short_msg = "Messages too short "
        if rd.target_id:
            short_msg += f"target lengths: {target_lengths}, "
        if rd.persuader_id:
            short_msg += f"persuader lengths: {persuader_lengths}"
        logger.warning(short_msg)

    if too_quick:
        quick_msg = "Messages sent too quickly "
        if rd.target_id:
            quick_msg += f"target times: {target_times}, "
        if rd.persuader_id:
            quick_msg += f"persuader times: {persuader_times}"
        logger.warning(quick_msg)

    return too_short, too_quick


def rounds_by_condition(
    session: Session, include_short_games: bool = False, group_by_persuader: bool = True
) -> dict[Condition, list[Tuple[Game, dict[str, Any]]]]:
    """
    Returns all of the current rounds as a dict of Conditions to (Games, metadata)
    Excludes unfinished rounds (where either of the target and persuader have not chosen)
    Excludes rounds in which one of the participants sent fewer than 10 characters per message
        or spent fewer than 5s on their turns (sent messages) on average.

        group_by_persuader (bool): If True groups games by persuader id, not target id
    """
    # Order the rounds first by persuader and then earliest to latest
    # Ignore incompleted rounds.
    rounds = session.exec(
        select(Round)
        .where(Round.persuader_choice.is_not(None))  # pylint: disable=no-member
        .where(Round.target_choice.is_not(None))  # pylint: disable=no-member
        .order_by(Round.persuader_id)
        .order_by(Round.updated_at)
    ).all()

    condition_to_games: dict[Condition, list[Game]] = {}
    for rd in rounds:
        game = Game(**rd.game_data)

        too_short, too_quick = exclude_round_messages(session, rd)
        if not game.game_over():
            logger.warning(f"Game for rd {rd} is not over")
        elif include_short_games or (not too_quick and not too_short):
            condition = rd.condition()

            # Set it so that just half of the condition has an id.
            condition = condition.as_non_id_role(no_target_id=group_by_persuader)

            if condition not in condition_to_games:
                condition_to_games[condition] = []

            logger.debug(game)

            condition_to_games[condition].append(game)
    return condition_to_games


def get_wait_statistics(session: Session, **_):
    """
    1) For each human participant and each completed round they play,
       computes how long they waited in the lobby before that round started.
       - For the Nth round (N>1):  round.created_at - (N-1).round.updated_at
       - For the 1st round:       round.created_at - participant.created_at
         (only if participant.created_at is set; else we skip it)

    2) For each completed round, computes the reply delays between the two sides,
       collecting two lists of delays (in seconds):
         a) including the very first inter‐message delay,
         b) excluding the very first inter‐message delay in each round.

    Prints summary (count, mean, median) for both metrics.
    """
    # 1) Fetch all completed rounds, sorted by creation time
    rounds = session.exec(select(Round).order_by(Round.created_at)).all()

    # Group per participant
    per_participant: dict[int, list[Round]] = defaultdict(list)
    for rd in rounds:
        for pid in (rd.persuader_id, rd.target_id):
            if pid is not None:
                per_participant[pid].append(rd)

    lobby_waits = []  # in seconds

    # Compute lobby waits
    for pid, player_rounds in per_participant.items():
        player = session.get(Participant, pid)
        # sort rounds by created_at
        player_rounds.sort(key=lambda r: r.created_at)
        for i, rd in enumerate(player_rounds):
            if rd.created_at is None:
                continue
            if i == 0:
                prev_ts = player.created_at
            else:
                # Subsequent rounds: measure from the time they last spoke in the prior round
                prev_rd = player_rounds[i - 1]
                # Determine whether they were target or persuader in the previous round
                is_target_prev = prev_rd.target_id == pid
                last_msg = get_last_sent_message(session, is_target_prev, prev_rd.id)
                if last_msg is None or last_msg.created_at is None:
                    # If they never sent a message, fall back to the previous round's updated_at
                    prev_ts = prev_rd.updated_at
                else:
                    prev_ts = last_msg.created_at
            if prev_ts and rd.created_at >= prev_ts:
                wait_t = (rd.created_at - prev_ts).total_seconds()
                lobby_waits.append(wait_t)

    # 2) Compute reply delays
    reply_incl = []
    reply_excl = []

    for rd in rounds:
        msgs = session.exec(
            select(SentMessage)
            .where(SentMessage.round_id == rd.id)
            .where(SentMessage.flagged == False)  # pylint: disable=singleton-comparison
            .order_by(SentMessage.created_at)
        ).all()

        last_ts = {True: None, False: None}  # True=target, False=persuader
        saw_first = False

        for msg in msgs:
            other = not msg.is_target
            prev = last_ts[other]
            if prev is not None and msg.created_at:
                delta_s = (msg.created_at - prev).total_seconds()
                reply_incl.append(delta_s)
                if saw_first:
                    reply_excl.append(delta_s)
                else:
                    saw_first = True
            last_ts[msg.is_target] = msg.created_at

    # Helper to print stats
    def summarize(name, data):
        if not data:
            print(f"{name}: no data")
            return
        print(
            f"{name}: count={len(data)}, "
            f"max={max(data):.1f}s, "
            f"mean={statistics.mean(data):.1f}s, "
            f"median={statistics.median(data):.1f}s"
        )

    print("\n=== Lobby Wait Times (seconds) ===")
    print("\t(This includes incomplete rounds.)")
    summarize("All inferred lobby waits", lobby_waits)

    print("\n=== Reply Delays (seconds) ===")
    summarize("Including first reply", reply_incl)
    summarize("Excluding first reply", reply_excl)


def handle_save_rounds(
    session, dry_run: bool = True, include_short_games: bool = True, **_
):
    """
    Handler to save rounds with and ouput
    """
    condition_to_games = rounds_by_condition(
        session, include_short_games=include_short_games
    )
    output_conditions_and_games(condition_to_games, dry_run=dry_run)


def handle_get_bonuses(session, hours: float | None = None, **_):
    """
    Handler to call get_bonuses and outuput
    """
    bonuses = get_bonuses(session, hours)
    output_bonuses_to_csv(bonuses)


def handle_get_feedback(session, **_):
    """
    Handler to call `get_feedback` with and ouput
    """
    feedback = get_feedback(session)
    pprint.pprint(feedback)


def output_bonuses_to_csv(bonuses: Counter[int]):
    """
    Outputs the bonuses as a CSV
    """
    bonuses_df = pd.DataFrame(
        list(bonuses.items()), columns=["Participant ID", "Bonus Count"]
    )
    print(bonuses_df)
    bonuses_df.to_csv("bonuses.csv", index=False)
    print("Bonuses saved to bonuses.csv.")


def print_finished_rounds(session: Session, external_id: str, **_):
    """
    For the passed external participant prints the number of rounds they finished.
    """
    base_statement = (
        select(func.count())  # pylint: disable=not-callable
        .select_from(ExternalUser)
        .join(Participant, ExternalUser.id == Participant.id)
        .where(ExternalUser.external_id == external_id)
    )

    as_target = (
        base_statement.join(Round, Round.target_id == Participant.id)
        .where(Round.persuader_choice.is_not(None))  # pylint: disable=no-member
        .where(Round.target_choice.is_not(None))  # pylint: disable=no-member
    )

    as_persuader = (
        base_statement.join(Round, Round.persuader_id == Participant.id)
        .where(Round.persuader_choice.is_not(None))  # pylint: disable=no-member
        .where(Round.target_choice.is_not(None))  # pylint: disable=no-member
    )

    count = session.exec(as_target).first() + session.exec(as_persuader).first()

    print(f"Completed rounds: {count}")


def get_feedback(session: Session) -> list[str]:
    """
    Returns the anonymized feedback given by each participant, if it exists.
    """

    base_statement = select(Participant.feedback).where(
        Participant.feedback != None  # pylint: disable=singleton-comparison
    )
    return session.exec(base_statement).all()


def get_bonuses(session: Session, hours: float | None = None) -> Counter[int]:
    """
    For each participant tabulates the number of rounds they have won
    for use in dispersing bonus payments.
    Only counts the participants whom have not yet been approved.
    A participant gets a bonus if they were the persuader and convinced
    the target to vote for their proposal or
    if they were the target and they chose the choice an ideal target would
    choose given perfect information.
    """

    base_statement = (
        select(ExternalUser.external_id, func.count())  # pylint: disable=not-callable
        .join(Participant, Participant.id == ExternalUser.id)
        .where(
            Participant.work_approved == None  # pylint: disable=singleton-comparison
        )
        .group_by(ExternalUser.external_id)
    )

    if hours is not None:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        base_statement = base_statement.where(Round.updated_at >= cutoff)

    persuader_statement = base_statement.join(
        Round, Round.persuader_id == Participant.id
    ).where(Round.persuader_choice == Round.target_choice)

    target_statement = base_statement.join(
        Round, Round.target_id == Participant.id
    ).where(
        and_(
            Round.target_perfect_info_choice  # pylint: disable=singleton-comparison
            != None,
            Round.target_perfect_info_choice == Round.target_choice,
        )
    )

    # Execute both queries and sum the results
    persuader_bonuses = session.exec(persuader_statement).all()
    target_bonuses = session.exec(target_statement).all()

    # Aggregate results in a single dictionary
    bonuses = Counter()
    bonuses.update(dict(persuader_bonuses))
    bonuses.update(dict(target_bonuses))

    return bonuses


def save_survey_responses(session: Session, **_):
    """Save survey responses to condition-specific folders."""

    rounds = session.exec(
        select(Round)
        .where(Round.persuader_choice.is_not(None))  # pylint: disable=no-member
        .where(Round.target_choice.is_not(None))  # pylint: disable=no-member
        .order_by(Round.updated_at)
    ).all()

    # Get current date for filename
    now = datetime.now().date().isoformat()

    condition_to_participants: dict[Condition, dict[int, Participant]] = {}

    for round_info in rounds:
        for participant_id in (round_info.target_id, round_info.persuader_id):
            if not participant_id:
                continue
            participant = session.get(Participant, participant_id)
            assert participant

            if (
                not participant.initial_survey_responses
                and participant.final_survey_responses
            ):
                logger.warning(
                    f"Participant {participant.id} did not complete both surveys"
                )
                continue

            condition = round_info.condition()
            if condition not in condition_to_participants:
                condition_to_participants[condition] = {}
            # This is a little wasteful as we loop through more rounds than we need
            # But by storing it in a set we only end up wtih unique participants
            condition_to_participants[condition][participant_id] = participant

    if not condition_to_participants:
        print("No completed surveys found")
        return

    for condition, id_to_participant in condition_to_participants.items():

        # Save survey data with date in filename
        all_surveys = []
        for participant in id_to_participant.values():
            for init_r, final_r in zip(
                participant.initial_survey_responses, participant.final_survey_responses
            ):
                assert init_r["statement"] == final_r["statement"]
                data = {
                    "participant_id": participant.id,
                    "role": participant.role,
                    "response_id": init_r["id"],
                    "statement": init_r["statement"],
                    "initial_rating": init_r["rating"],
                    "final_rating": final_r["rating"],
                }
                all_surveys.append(data)

        # Save it in the same directory
        dir_name = condition_to_dir(condition)

        df = pd.DataFrame(all_surveys)
        results_dir = os.path.join(RESULTS_DIR, dir_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        file_name = os.path.join(results_dir, f"completed-surveys_{now}.csv")

        print(f"Saving condition surveys to {file_name}")

        df.to_csv(file_name, index=False)


def get_last_sent_message(
    session: Session, is_target: bool, round_id: int
) -> SentMessage | None:
    """
    Returns the last message sent by target or persuader in `round_id` using the `session` or None.
    """
    statement = (
        select(SentMessage)
        .where(SentMessage.round_id == round_id)
        .where(SentMessage.is_target == is_target)
        .order_by(SentMessage.created_at.desc())  # pylint: disable=no-member
    )
    return session.exec(statement).first()


def populate_tables(
    session: Session, difficulty_conditions: Iterable[str], non_mental: bool = False
):
    """
    Populates the scenario and model tables from files.

    Parameters:
    difficulty_conditions: Iterable[str] -- the payoff model types to load
    """

    ## Add entries for the GameModel tables
    for game_model_type in difficulty_conditions:
        non_solutions = game_model_type != "solution"
        difficulty = game_model_type if non_solutions else None
        payoff_file = get_payoff_file_path(
            non_solutions=non_solutions, difficulty=difficulty
        )
        df = pd.read_json(payoff_file, orient="records", lines=True)
        df_randomized = df.sample(frac=1).reset_index(drop=True)
        for _, row in df_randomized.iterrows():
            model = Model(data=row.to_dict(), game_model_type=game_model_type)
            session.add(model)

    ## Add entries for the different scenario types
    if non_mental:
        df = pd.read_json(NON_MENTAL_SCENARIOS_FILE, lines=True)
    else:
        df = pd.read_json(SCENARIOS_FILE, lines=True)
    for _, row in df.iterrows():
        # Check if scenario already exists
        existing_scenario = session.get(Scenario, row["id"])
        if not existing_scenario:
            scenario = Scenario(**row.to_dict())
            session.add(scenario)
    session.commit()


def get_participant_rounds(
    participant: Participant, session: Session
) -> list[Round] | None:
    """
    For the given participant, returns, using `session`, a list of rounds which the participant
    appears in or None
    """
    statement = select(Round).where(
        or_(
            Round.persuader_id == participant.id,
            Round.target_id == participant.id,
        )
    )
    return session.exec(statement).all()


def get_paired_participant(
    participant: Participant,
    session: Session,
    settings: ServerSettings,
) -> Participant | None:
    """
    Returns a participant for this `participant` to play with, using `session`.
    If `settings.participants_rematch` allows participants to be paired repeatedly.
    Only returns participants who have been in the waiting room for less than
    `settings.waiting_room_timeout`.
    """
    if not participant.entered_waiting_room:
        return None

    n_minutes_ago = datetime.now(timezone.utc) - settings.waiting_room_timeout

    chosen_participant = None

    # Make the relevant sql query
    statement = (
        select(Participant)
        .where(Participant.id != participant.id)
        .where(
            Participant.entered_waiting_room  # pylint: disable=singleton-comparison
            != None
        )
        .where(Participant.entered_waiting_room >= n_minutes_ago)
        .with_for_update(skip_locked=True)
    )

    if participant.role and participant.role != "either":
        if participant.role == "persuader":
            statement = statement.where(
                or_(
                    Participant.role == None,  # pylint: disable=singleton-comparison
                    Participant.role != "persuader",
                )
            )
        else:  # participant.role == 'target'
            statement = statement.where(
                or_(
                    Participant.role == None,  # pylint: disable=singleton-comparison
                    Participant.role != "target",
                )
            )
    if not settings.participants_rematch:
        targets = (
            select(Round.persuader_id)
            .where(Round.persuader_id != None)  # pylint: disable=singleton-comparison
            .where(Round.target_id == participant.id)
        )
        persuaders = (
            select(Round.target_id)
            .where(Round.target_id != None)  # pylint: disable=singleton-comparison
            .where(Round.persuader_id == participant.id)
        )
        statement = statement.where(
            Participant.id.not_in(targets)  # pylint: disable=no-member
        ).where(
            Participant.id.not_in(persuaders)  # pylint: disable=no-member
        )

    # The other participant must be in the same condition as this participant, if assigned
    if participant.targets_values is not None:
        statement = statement.where(
            or_(
                (Participant.targets_values == participant.targets_values),
                (Participant.targets_values.is_(None)),  # pylint: disable=no-member
            )
        )

    if participant.reveal_motivation is not None:
        statement = statement.where(
            or_(
                (Participant.reveal_motivation == participant.reveal_motivation),
                (Participant.reveal_motivation.is_(None)),  # pylint: disable=no-member
            )
        )

    if participant.reveal_belief is not None:
        statement = statement.where(
            or_(
                (Participant.reveal_belief == participant.reveal_belief),
                (Participant.reveal_belief.is_(None)),  # pylint: disable=no-member
            )
        )

    if participant.allow_lying is not None:
        statement = statement.where(
            or_(
                (Participant.allow_lying == participant.allow_lying),
                (Participant.allow_lying.is_(None)),  # pylint: disable=no-member
            )
        )

    participants = session.exec(statement).all()

    if participants:

        for other_participant in participants:
            # Is there any overlap in the games both participants want to play?
            if Counter(participant.game_model_types_remaining) & Counter(
                other_participant.game_model_types_remaining
            ):
                chosen_participant = other_participant
    return chosen_participant


def round_types_count(session: Session) -> TypeCounter[Condition]:
    """
    Returns a `Counter` over all of the seen `Condition`s.
    """
    # NB: we only count completed games which could result in too many games of a certain type
    # but that isn't so bad

    # Select the boolean expressions and the model names, and count the occurrences
    columns = (
        Round.persuader_id,
        Round.target_id,
        Round.llm_persuader,
        Round.llm_target,
        Round.targets_values,
        Round.allow_lying,
        Round.reveal_belief,
        Round.reveal_motivation,
    )
    statement = (
        select(*columns)
        .where(Round.persuader_choice.is_not(None))  # pylint: disable=no-member
        .where(Round.target_choice.is_not(None))  # pylint: disable=no-member
    )
    results = session.exec(statement).all()

    all_conditions = []
    for (
        persuader_id,
        target_id,
        llm_persuader,
        llm_target,
        targets_values,
        allow_lying,
        reveal_belief,
        reveal_motivation,
    ) in results:
        condition = Condition(
            roles=Roles(
                human_persuader=persuader_id is not None,
                human_target=target_id is not None,
                llm_persuader=llm_persuader,
                llm_target=llm_target,
            ),
            targets_values=targets_values,
            allow_lying=allow_lying,
            reveal_belief=reveal_belief,
            reveal_motivation=reveal_motivation,
        )
        all_conditions.append(condition)

    current_count = Counter(all_conditions)
    return current_count


#### API helpers


def get_round_types_remaining(
    session: Session, condition_num_rounds: TypeCounter[Condition]
) -> (set[Condition], set[Condition]):
    """
    In the given `session`, tabulates the remaining round types given the rounds played
    (e.g. from `condition_num_rounds`).
    Returns
    - set[Condition]: the Conditions yet to be filled for paired rounds
    - set[Condition]: the Conditions yet to be filled for non-paired rounds
    """
    current_count = round_types_count(session)
    remaining_round_counts = condition_num_rounds - current_count
    non_paired_conditions = set()
    paired_conditions = set()

    for condition in remaining_round_counts.keys():
        if condition.is_paired_human():
            paired_conditions.add(condition)
        else:
            non_paired_conditions.add(condition)

    return paired_conditions, non_paired_conditions


def choose_condition(
    participant: Participant,
    chosen_participant: Participant,
    paired_rounds_remaining_set: set[Condition],
    non_paired_rounds_remaining_set: set[Condition],
    overassign_non_paired_conditions: bool,
    enforce_player_round_condition: bool,
    condition_num_rounds: TypeCounter[Condition],
    waiting_room_timeout: timedelta,
) -> Condition:
    """
    Chooses the condition to slot this `participant` into, based on the current counts of
    completed rounds. Returns the name of the condition type.
    Returns None if no condition chosen (the participant should wait)
    """
    if not participant.entered_waiting_room:
        raise ValueError("Participant must be in waiting room")

    chosen_condition = None

    participant_waiting = participant.waiting_time()

    if len(non_paired_rounds_remaining_set) == 0 and not paired_rounds_remaining_set:
        raise ValueError("No more rounds to assign")

    if participant.round_condition:
        # If we can, have this player play this condition
        chosen_condition = Condition(**participant.round_condition)

        # Tell them to wait b/c no participant
        if chosen_condition.is_paired_human() and not chosen_participant:
            chosen_condition = None

    elif chosen_participant:
        # Either choose the other participant's condition or a random one
        # if neither are assigned
        if chosen_participant.round_condition:
            chosen_condition = Condition(**chosen_participant.round_condition)
        else:
            chosen_condition = random.choice(list(paired_rounds_remaining_set))

    elif (
        not paired_rounds_remaining_set or participant_waiting > waiting_room_timeout
    ) and (overassign_non_paired_conditions or non_paired_rounds_remaining_set):
        # Either there are no more paired rounds or the participant has been waiting too long
        # and there are non paired rounds left or we will overstuff non paired rounds

        # Even if we have filled all the slots, we simply add more participants
        # Make sure not to assign to a paired human role from the main condition
        non_paired_roles = set(
            filter(lambda c: not c.is_paired_human(), condition_num_rounds.keys())
        )

        if len(non_paired_rounds_remaining_set) > 0:
            # We have not yet filled the necessary slots
            non_paired_roles = non_paired_rounds_remaining_set

        possible_roles = list(
            filter(
                lambda condition: (not participant.role)
                or (participant.role == "either")
                or (condition.roles.human_target and participant.role == "target")
                or (
                    condition.roles.human_persuader and participant.role == "persuader"
                ),
                non_paired_roles,
            )
        )

        # TODO: the below really should not happen
        if len(possible_roles) < 1:
            raise ValueError("No roles left to assign")

        chosen_condition = random.choice(possible_roles)

    # Change the participants to relfect the condition
    if enforce_player_round_condition and chosen_condition:
        if participant.round_condition:
            assert Condition(**participant.round_condition) == chosen_condition
        else:
            participant.round_condition = chosen_condition.model_dump()

        if chosen_participant:
            if chosen_participant.round_condition:
                assert (
                    Condition(**chosen_participant.round_condition) == chosen_condition
                )
            else:
                chosen_participant.round_condition = chosen_condition.model_dump()

    return chosen_condition
