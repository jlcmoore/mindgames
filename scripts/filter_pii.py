"""
Author: Jared Moore
Date: January, 2025

Contains scripts to 
"""

import argparse
import json
import logging

from modelendpoints.query import Endpoint, Messages

from experiments.utils import load_game_results
from mindgames.utils import replace_json_chars

logger = logging.getLogger(__name__)


PII_PROMPT = """\
Your job is to figure out if *any* of the messages we give you \
contain personally identifiable information (PII)---\
such as a name, address, birthday, etc. \
Most messages will not contain any PII, but air on the cautious side.

Format your response as a JSON list (the same as we give you), \
simply repeating back *only* the messages which contain PII.

E.g.:
```
[
    {{'role' : '<message role>', 'content' : '<message content>'}},
]
```

Input Messages:
```
{messages}
```
"""


def contains_pii(messages: Messages) -> Messages:
    """
    Queries an LLM about the given messages to see if they contain PII.

    Parameters:
    messages (str OR list[dict[str, str]]): The messages

    Returns:
    Messages: the messages in the list which contain PII, or []
    """
    if not messages:
        return []

    messages_formatted = json.dumps(messages, indent=4)
    this_prompt = PII_PROMPT.format(messages=messages_formatted)

    query_messages = [{"role": "user", "content": this_prompt}]
    with Endpoint(
        source="openai", temperature=0, model="gpt-4o", max_tokens=512
    ) as endpoint:
        response = endpoint(messages=query_messages)

    result = []
    response_text = replace_json_chars(response["text"])
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as err:
        logger.error(f"Could not decode JSON response, {response_text}")
        logger.error(err)

    # Validation of output
    input_tuple = set(tuple(msg.items()) for msg in messages)
    validated_pii = []
    for pii_message in result:
        if tuple(pii_message.items()) in input_tuple:
            validated_pii.append(pii_message)
        else:
            logger.error(f"PII message, {pii_message}, not found in input.")
    return validated_pii


def main():
    parser = argparse.ArgumentParser(description="Process date argument.")
    parser.add_argument("date", type=str, help="Date in the format YYYY-MM-DD")
    args = parser.parse_args()

    conditions_to_games = load_game_results(args.date)

    # Just focusing on the rational target conditions for now.
    rational_conditions = list(
        filter(
            lambda c: c.is_rational_target()
            and (c.roles.human_target or c.roles.human_persuader),
            list(conditions_to_games.keys()),
        )
    )
    desired_conditions = rational_conditions

    # games_with_pii: list[Tuple[Game, Messages]] = []
    for condition in desired_conditions:
        list_of_games = conditions_to_games[condition]
        for games in list_of_games:
            for game in games:
                pii_messages = contains_pii(game.messages)
                if pii_messages:
                    print(pii_messages)
                    print()
                    print()
                    # games_with_pii.append()


if __name__ == "__main__":
    main()
