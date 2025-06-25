"""
Author: Jared Moore
Date: April, 2025

Utilities to read the sql database
"""

import argparse
from collections import defaultdict
import logging
import statistics


from sqlalchemy import inspect
from sqlmodel import Session, select, create_engine

import mindgames
from mindgames.model import GameModel
from mindgames.utils import set_logger

from .current_round_helpers import filter_models_for_target
from .sql_model import (
    Participant,
    Scenario,
    Model,
    SQLITE_FILE_NAME,
    SQLITE_URL_FMT,
    CONNECT_ARGS,
)

from .sql_queries import (
    handle_save_rounds,
    handle_get_bonuses,
    handle_get_feedback,
    print_finished_rounds,
    save_survey_responses,
    get_wait_statistics,
)

logger = logging.getLogger(__name__)


def read_database():
    """
    Read the database main function.
    """
    parser = argparse.ArgumentParser(prog="Database Queries")
    parser.add_argument(
        "--database",
        type=str,
        default=SQLITE_FILE_NAME,
        help="Path to the database",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="WARNING",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="subcommand help", required=True
    )

    save_rounds_parser = subparsers.add_parser(
        "save-rounds", help="Save the current rounds"
    )
    save_rounds_parser.set_defaults(func=handle_save_rounds)
    save_rounds_parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Whether not to output the results to a file",
    )
    save_rounds_parser.add_argument(
        "--include-short-games",
        default=False,
        action="store_true",
        help="Whether to include the games that are too short",
    )

    gen_bonuses_parser = subparsers.add_parser(
        "get-bonuses", help="Outputs bonuses as csv"
    )
    gen_bonuses_parser.set_defaults(func=handle_get_bonuses)
    gen_bonuses_parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Only include rounds completed in the last N hours (default: all time)",
    )

    feedback_parser = subparsers.add_parser("get-feedback", help="Print all feedback")
    feedback_parser.set_defaults(func=handle_get_feedback)

    print_finished = subparsers.add_parser(
        "print-completed", help="Print completed rounds"
    )
    print_finished.add_argument("--external-id", type=str, required=True)
    print_finished.set_defaults(func=print_finished_rounds)

    save_surveys_parser = subparsers.add_parser(
        "save-surveys", help="Save survey responses"
    )
    save_surveys_parser.set_defaults(func=save_survey_responses)

    wait_times_parser = subparsers.add_parser(
        "wait-times", help="Compute wait statistics"
    )
    wait_times_parser.set_defaults(func=get_wait_statistics)

    survey_stats_parser = subparsers.add_parser(
        "survey-stats", help="Compute survey statistics"
    )
    survey_stats_parser.set_defaults(func=participant_surveys)

    args = parser.parse_args()

    set_logger(args.log, logger=logger)
    filename = SQLITE_URL_FMT.format(filename=args.database)
    engine = create_engine(filename, echo=False, connect_args=CONNECT_ARGS)
    inspector = inspect(engine)
    with Session(engine) as session:
        expected_tables = {
            "model",
            "scenario",
            "round",
            "externaluser",
            "participant",
            "sentmessage",
        }
        existing_tables = set(inspector.get_table_names())
        if expected_tables != existing_tables:
            raise ValueError("Necessary tables do not exist.")

        all_args = vars(args).copy()
        all_args.pop("command", None)
        args.func(session, **all_args)


def participant_surveys(session: Session, **_):
    """
    Outputs descriptive statistics about the initial surveys
    and how often participants’ stated preferences match
    any 'solution' GameModel for each scenario.
    """
    # 1) Load participants with an initial survey
    participants = session.exec(
        select(Participant).where(
            Participant.initial_survey_responses.is_not(  # pylint: disable=no-member
                None
            )
        )
    ).all()
    if not participants:
        print("No participants with initial surveys found.")
        return

    # 2) Load all 'solution' GameModels from the Model table
    models = session.exec(
        select(Model).where(Model.game_model_type == "solution")
    ).all()
    game_models = [GameModel(**m.data) for m in models]

    # 3) Load all scenarios
    scenarios = session.exec(select(Scenario)).all()

    # --- Part B: Literal survey questions — mean rating per statement ---
    stmt_to_ratings: dict[str, list[int]] = defaultdict(list)
    for p in participants:
        for resp in p.initial_survey_responses:
            stmt = resp["statement"]
            rating = resp.get("rating")
            # skip if rating missing
            if rating is not None:
                stmt_to_ratings[stmt].append(rating)

    print("\n\n=== Initial Survey: Mean Rating per Question ===")
    for stmt, ratings in stmt_to_ratings.items():
        print(
            f"{stmt!r}\n   count={len(ratings):3d}"
            f"\tmean={statistics.mean(ratings):4.2f}"
            f"\tmedian={statistics.median(ratings):4.2f}\n"
        )

    # --- Part C: Scenario - Model matching ---
    total_participants = len(participants)
    print("\n=== Scenario - Model Match Rates ===")
    scenario_rates: dict[str, float] = {}
    for scenario in scenarios:
        match_count = 0
        for p in participants:
            sc_copy = mindgames.utils.Scenario(**scenario.model_dump())
            if filter_models_for_target(game_models, p, sc_copy):
                match_count += 1

        rate = match_count / total_participants
        scenario_rates[scenario.id] = rate
        print(
            f"Scenario {scenario.id:10s}: "
            f"{match_count:3d}/{total_participants:3d}  {rate*100:5.1f}%"
        )

    # overall average match rate
    if scenario_rates:
        avg_rate = statistics.mean(scenario_rates.values())
        print(f"\nOverall average match rate: {avg_rate*100:5.1f}%")
    else:
        print("\nNo scenarios found, skipping match-rate summary.")
