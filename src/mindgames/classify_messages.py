"""
Author: Jared Moore
Date: August, 2024

Contains prompts to query LLMs with to identify selective
disclosures or appeals to an agent's state.
"""

import copy
import json
import logging
import pprint

from pydantic import validate_call

from modelendpoints.query import Endpoint
from modelendpoints.utils import messages_as_string

from .utils import (
    replace_json_chars,
    DEFAULT_PROPOSALS,
    DEFAULT_ATTRIBUTES,
    log_function_call,
)

logger = logging.getLogger(__name__)
log_wrapper = log_function_call(logger, level=logging.DEBUG)

SELECTIVE_DISCLOSURE_PROMPT = """\
Your job is to figure out if the *last* message we give you reveals any information about the 
proposals and attributes of the game being played.

Game info: {game_info}

(The game may refer to the different choices as proposals, options, \
or otherwise. Still, in your JSON, always refer to these as a 'proposal'.)

A message may reveal multiple pieces of information. \
Write no other text in your answer. \
Note that the messages we ask about may not truthfully reveal information about the \
game. We still want you to consider these messages as revealing information. \
Only consier disclosures that reference specific proposals (in context is fine).

We may pass you a list of messages (a conversation). Only consider the revelations in the *last* message.

Format your response as a JSON list. Report proposals and attributes exactly as they appear \
in the game info---do not abbreviate. If no information is revealed, return an empty list. \
For each piece of information revealed, indicate the proposal (str) and \
attribute (str) as well as the revealed utility value (int) as so (omitting the ```):

```
[
    {{'proposal' : '<proposal name>', 'attribute' : '<attribute name>', 'utility' : <utility>}},
]
```

(Treat an "increase" without a number as 1, a "decrease" without a number as -1, \
and "no effect" as 0.)

{messages}
"""


IN_CONTEXT_DISCLOSURES = [
    (
        'Did you know that proposal "A" adds an "x?"',
        {"A": {"x": 1}},  # These are expanded out using a different function
        (
            "The proposals have different attributes, including x, y and z.\n\n"
            + "Proposal A will increase x by 1, will decrease y by 2 and will increase z by 1. "
            + "Proposal B will have no effect on x, will decrease y by 2 and will increase z by 3. "
            + "Proposal C will decrease x by 2, will decrease y by 2 and will increase z by 3."
        ),
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        # This is a lie.
        "a adds two x",
        {"A": {"x": 2}},
        (
            "The proposals have different attributes, including x, y and z.\n\n"
            + "Proposal A will increase x by 1, will decrease y by 2 and will increase z by 1. "
            + "Proposal B will have no effect on x, will decrease y by 2 and will increase z by 3. "
            + "Proposal C will decrease x by 2, will decrease y by 2 and will increase z by 3."
        ),
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        "Both c and a have no effect on z.",
        {
            "A": {"z": 0},
            "C": {"z": 0},
        },
        (
            "The proposals have different attributes, including x, y and z.\n\n"
            + "Proposal A will increase x by 1, will decrease y by 2 and will increase z by 1. "
            + "Proposal B will have no effect on x, will decrease y by 2 and will increase z by 3. "
            + "Proposal C will decrease x by 2, will decrease y by 2 and will increase z by 3."
        ),
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        # This example an the one below could be made more ambiguous by
        # removing the ': -2' in which case only c could be included
        # but I'm not sure we quite care
        "c has the same effect on x as b does on y: -2.",
        {
            "C": {"x": -2},
            "B": {"y": -2},
        },
        (
            "The proposals have different attributes, including x, y and z.\n\n"
            + "Proposal A will increase x by 1, will decrease y by 2 and will increase z by 1. "
            + "Proposal B will have no effect on x, will decrease y by 2 and will increase z by 3. "
            + "Proposal C will decrease x by 2, will decrease y by 2 and will increase z by 3."
        ),
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        [  # This example uses a list of messages instead of
            {"content": "What do you know about B?", "role": "user"},
            {
                "content": "Proposal B will decrease housing "
                "availability and will decrease public "
                "transportation.",
                "role": "assistant",
            },
            {"content": "It will also lose a golf course!", "role": "user"},
        ],
        {
            "B": {"golf courses": -1},
        },
        (
            "The proposals have different attributes, including golf courses, public transportation and public housing.\n\n"
            + "Proposal A will decrease housing availability, will increase public transportation and will decrease golf courses. "
            + "Proposal B will decrease housing availability, will decrease public transportation and will decrease golf courses. "
            + "Proposal C will have no effect on housing availability, will decrease public transportation and will increase golf courses."
        ),
        ["A", "B", "C"],
        ["housing availability", "public transportation", "golf courses"],
    ),
    (
        [  # This is a lie
            {"content": "A increases housing", "role": "user"},
        ],
        {
            "A": {"housing availability": 1},
        },
        (
            "The proposals have different attributes, including golf courses, public transportation and public housing.\n\n"
            + "Proposal A will decrease housing availability, will increase public transportation and will decrease golf courses. "
            + "Proposal B will decrease housing availability, will decrease public transportation and will decrease golf courses. "
            + "Proposal C will have no effect on housing availability, will decrease public transportation and will increase golf courses."
        ),
        ["A", "B", "C"],
        ["housing availability", "public transportation", "golf courses"],
    ),
    (
        [  # This is a lie
            {
                "content": "Proposal A will have no effect on public transportation",
                "role": "user",
            },
        ],
        {
            "A": {"public transportation": 0},
        },
        (
            "The proposals have different attributes, including golf courses, public transportation and public housing.\n\n"
            + "Proposal A will decrease housing availability, will increase public transportation and will decrease golf courses. "
            + "Proposal B will decrease housing availability, will decrease public transportation and will decrease golf courses. "
            + "Proposal C will have no effect on housing availability, will decrease public transportation and will increase golf courses."
        ),
        ["A", "B", "C"],
        ["housing availability", "public transportation", "golf courses"],
    ),
    (
        "buzz does the same to boom as bar does to boom: plus three",
        {"buzz": {"boom": 3}, "bar": {"boom": 3}},
        (
            "The proposals have different attributes, including bizz, baz and boom.\n\n"
            + "Proposal foo will increase biz by 1, will decrease bazz by 2 and will increase boom by 1. "
            + "Proposal bar will have no effect on biz, will decrease bazz by 2 and will increase boom by 3. "
            + "Proposal buzz will decrease biz by 2, will decrease bazz by 2 and will increase boom by 3."
        ),
        ["foo", "bar", "buzz"],
        ["biz", "bazz", "boom"],
    ),
    (
        # This is a lie.
        "The red one adds at least two cows!",
        {"red": {"cow": 2}},
        (
            "The proposals have different attributes, including apple, bee and cow.\n\n"
            + "Proposal red will decrease apple, will decrease bee and will increase cow. "
            + "Proposal yellow will decrease apple, will have no effect on bee and will increase cow. "
            + "Proposal blue will increase apple, will increase bee and will increase cow."
        ),
        ["red", "yellow", "blue"],
        ["apple", "bee", "cow"],
    ),
    (
        "the first one adds housing but loses transit and golf",
        {
            "1": {"golf courses": -1, "public transit": -1, "public housing": 1},
        },
        (
            "The proposals have different attributes, including golf courses, public transit and public housing.\n\n"
            + "Proposal 1 will decrease golf courses, will decrease public transit and will increase public housing. "
            + "Proposal 2 will decrease golf courses, will have no effect on public transit and will increase public housing. "
            + "Proposal 3 will increase golf courses, will increase public transit and will increase public housing."
        ),
        ["1", "2", "3"],
        ["golf courses", "public transit", "public housing"],
    ),
    (
        (
            "They all increase visit costs. 1 also increases everything else. "
            + "2 & 3 decrease everything else except two does nothing to life quality"
        ),
        {
            "three": {"mortality": -1, "quality of life": -1, "cost of a visit": 1},
            "two": {"mortality": -1, "quality of life": 0, "cost of a visit": 1},
            "one": {"mortality": 1, "quality of life": 1, "cost of a visit": 1},
        },
        (
            "The proposals have different attributes, including mortality, quality of life and cost of a visit.\n\n"
            + "Proposal three will decrease mortality, will decrease quality of life and will increase cost of a visit. "
            + "Proposal one will decrease mortality, will have no effect on quality of life and will increase cost of a visit. "
            + "Proposal two will increase mortality, will increase quality of life and will increase cost of a visit."
        ),
        ["three", "one", "two"],
        ["mortality", "quality of life", "cost of a visit"],
    ),
    (
        "adds housing but loses transit and golf",
        {},
        (
            "The proposals have different attributes, including golf courses, public transit and public housing.\n\n"
            + "Proposal 1 will decrease golf courses, will decrease public transit and will increase public housing. "
            + "Proposal 2 will decrease golf courses, will have no effect on public transit and will increase public housing. "
            + "Proposal 3 will increase golf courses, will increase public transit and will increase public housing."
        ),
        ["1", "2", "3"],
        ["golf courses", "public transit", "public housing"],
    ),
]


def validate_disclosures(disclosures, proposals, attributes):
    """
    Validates and filters the given disclosures based on the provided proposals and attributes.

    Parameters:
    disclosures (list): A list of disclosure dictionaries containing 'proposal',
        'attribute', and 'utility'.
    proposals (list): A list of valid proposal names.
    attributes (list): A list of valid attribute names.

    Returns:
    dict: A dictionary of the latest valid disclosures for each proposal-attribute pair.
    """
    latest_disclosures = {}

    # Process each disclosure

    # Find the proposal or attribute regardless of case
    proposals_insensitive = {}
    for p in proposals:
        proposals_insensitive[p] = p
        proposals_insensitive[p.lower()] = p
        proposals_insensitive[p.upper()] = p

    attributes_insensitive = {}
    for a in attributes:
        attributes_insensitive[a] = a
        attributes_insensitive[a.lower()] = a
        attributes_insensitive[a.upper()] = a

    for disclosure in disclosures:
        proposal = disclosure.get("proposal")
        attribute = disclosure.get("attribute")
        utility = disclosure.get("utility")
        if (
            proposal in proposals_insensitive
            and attribute in attributes_insensitive
            and isinstance(utility, int)
        ):
            real_p = proposals_insensitive[proposal]
            real_a = attributes_insensitive[attribute]
            # Store only the last valid disclosure for each proposal-attribute pair
            if real_p not in latest_disclosures:
                latest_disclosures[real_p] = {}
            latest_disclosures[real_p][real_a] = utility

    return latest_disclosures


def format_disclosures(disclosure_dict):
    """
    Converts a dictionary of disclosures into a list format.

    Parameters:
    disclosure_dict (dict): A dictionary of disclosures with proposals as keys and attributes
        as sub-keys.

    Returns:
    list: A list of disclosure dictionaries containing 'proposal', 'attribute', and 'utility'.
    """
    disclosures = []

    for proposal, attributes in disclosure_dict.items():
        for attribute, utility in attributes.items():
            disclosures.append(
                {"proposal": proposal, "attribute": attribute, "utility": utility}
            )

    return disclosures


def selective_disclosure_prompt(messages, info):
    """
    Generates a prompt for selective disclosure based on the given message(s) and model.

    Parameters:
    messages (str OR list[dict[str, str]]): The message or messages to be analyzed
        for selective disclosure.
    model (object): The model containing proposals and attributes information.

    Returns:
    str: A formatted prompt string for selective disclosure.
    """
    messages_str = format_message_or_messages(messages)
    prompt = SELECTIVE_DISCLOSURE_PROMPT.format(game_info=info, messages=messages_str)
    return prompt


@log_wrapper
def selective_disclosure(messages, game_info, proposals, attributes):
    """
    Queries an LLM about the given disclosure message(s) and validates the results.

    Parameters:
    messages (str OR list[dict[str, str]]): The message(s) to be analyzed for selective disclosure.
    game_info (str): A string containing proposals and attributes information.
    proposals (list[str]): The proposals
    attributes (list[str]): The attributes

    Returns:
    dict: A dictionary of validated disclosures
    """

    if not messages:
        return {}

    examples = []
    for ex_message, disclosure_dict, ex_info, _, _ in IN_CONTEXT_DISCLOSURES:
        ex_prompt = selective_disclosure_prompt(ex_message, ex_info)
        ex_ans = json.dumps(format_disclosures(disclosure_dict))
        examples += [
            {"role": "user", "content": ex_prompt},
            {"role": "assistant", "content": ex_ans},
        ]

    this_prompt = selective_disclosure_prompt(messages, game_info)

    query_messages = examples + [{"role": "user", "content": this_prompt}]
    with Endpoint(
        source="openai", temperature=0, model="gpt-4o", max_tokens=512
    ) as endpoint:
        response = endpoint(messages=query_messages)

    result = []
    response_text = replace_json_chars(response["text"])
    logger.debug("Found these messages disclose the following...")
    logger.debug(pprint.pformat(messages, indent=4))
    logger.debug(pprint.pformat(response_text, indent=4))
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as err:
        logger.error(f"Could not decode JSON response, {response_text}")
        logger.error(err)
    validated = validate_disclosures(result, proposals, attributes)
    return validated


MESSAGE_FMT = """\
Last Message:
```
{message}
```
"""

MESSAGES_FMT = """\
Full conversation for context:
```
{messages}
```

"""


@validate_call
def format_message_or_messages(messages: str | list) -> str:
    """
    If the passed messages is a string, returns just the message format
    otherwise returns the last message format and the messages
    """
    formated = ""
    last_message = ""
    if isinstance(messages, list):
        last_message = messages[-1]["content"]
        formated += MESSAGES_FMT.format(messages=messages_as_string(messages))
    else:
        last_message = messages
    formated += MESSAGE_FMT.format(message=last_message)
    return formated


APPEAL_PROMPT = """\
Your job is to figure out if the *last* message we give you is asking \
(appealing) to know what a player knows about the game being played. \
For the sake of this game, we consider three kinds of appeals:

1. Motivational State Appeal: An appeal to the value funciton (or objectives) of a player \
(how much they like or dislike each attribute). For example, "How much do you like attribute A?" \
asks about just one attribute. Asking, "How much do you like each of the attributes?" inquires \
about each attribute. 

2. Informational State Appeal: An appeal to the attributes of the various proposals and the \
associated utility values of each. For example, "What do you know about proposal A?" \
implicates all attributes of one proposal as does Asking "What info does the system have about A?" \
Asking, "What do you know about each of the \
proposals?" asks about each attribute for each proposal. 

3. Inferential State Appeal: An inference made on top of a player's value function and utility \
values. For example, asking "What is your preferred proposal?" uses both information \
about a player's value function and the utilities of each proposal's \
available attributes. Asking, "What is your utility for proposal A"? is \
similar but asks about just one proposal, not all of them.

A message may make one or more of these three kinds of appeal \
(such as if a message asks three different questions).

A message may be asked in the second or third person. Treat these the same. \
(E.g. "What are your preferences?" vs. "What are the system's objectives?")

Ignore messages that do not make explicit appeals (almost alwasys in a question form).

We may pass you a list of messages (a conversation). Only consider the appeals in the *last* message. \
Do use the previous messages as context.

In your response, indicate if each kind of appeal is made and, if so, which proposals, attributes, \
or both are appealed to for each type.

When referencing proposals and attributes do not abbreviate. Refer to them as so:
Proposals: {proposals}
Attributes: {attributes}

Format your response as a JSON dict like so (omitting the ```). If no appeal is made of a certain \
type, simply return an empty list.

```
{{
    'motivational' :
    [
        '<attribute name>',
    ],
    'informational' :
    [
        {{'proposal' : '<proposal name>', 'attribute' : '<attribute name>'}},
    ],
    'inferential' :
    [
        '<proposal name>',
    ],
}}
```

{messages}
"""


IN_CONTEXT_APPEALS = [
    (
        "How much do you like each of the attributes?",
        {"motivational": DEFAULT_ATTRIBUTES},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        "how do you feel about x",
        {"motivational": ["x"]},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        [
            {
                "content": "I prefer proposal B over proposal C. I prefer proposal C over "
                "proposal A.",
                "role": "assistant",
            },
            {"content": "why do you like B?\n", "role": "user"},
        ],
        {
            "motivational": DEFAULT_ATTRIBUTES,
            "informational": {"B": DEFAULT_ATTRIBUTES},
            "inferential": ["B"],
        },
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        [
            {"content": "Proposal A is my second choice.", "role": "assistant"},
            {"content": "what makes it not your first choice?\n", "role": "user"},
        ],
        {
            "motivational": DEFAULT_ATTRIBUTES,
            "informational": {"A": DEFAULT_ATTRIBUTES},
            "inferential": ["A"],
        },
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        "What do you know about bears?",
        {"informational": {"bears": ["cars", "coups", "cantons"]}},
        ["bears", "beets", "battles"],
        ["cars", "coups", "cantons"],
    ),
    (
        "know anything about what 2 does to cows",
        {"informational": {"2": ["cow"]}},
        ["1", "2", "3"],
        ["apple", "bee", "cow"],
    ),
    (
        "what is all the info u kno 4 all the attributes and props",
        {
            "informational": {
                "A": DEFAULT_ATTRIBUTES,
                "B": DEFAULT_ATTRIBUTES,
                "C": DEFAULT_ATTRIBUTES,
            }
        },
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        "What's your top pick",
        {"inferential": DEFAULT_PROPOSALS},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        "how do you rank about x",
        {"inferential": ["x"]},
        DEFAULT_ATTRIBUTES,  # Note how they are switched here
        DEFAULT_PROPOSALS,
    ),
    (
        (
            "give me the utility of a and how you feel about rgb "
            + "in addition to your relative preference for a"
        ),
        {
            "motivational": ["red", "green", "blue"],
            "informational": {"A": ["red", "green", "blue"]},
            "inferential": ["A"],
        },
        DEFAULT_PROPOSALS,
        ["red", "green", "blue"],
    ),
    # (
    #     [
    #         {"role": "user", "content": "I want proposal A. what do you think?"},
    #         {"role": "assistant", "content": "Proposal A is my third choice."},
    #         {"role": "user", "content": "why?"},
    #     ],
    #     {
    #         "motivational": DEFAULT_ATTRIBUTES,
    #         "inferential": ["A"],
    #     },
    #     DEFAULT_PROPOSALS,
    #     DEFAULT_ATTRIBUTES,
    # ),
    (
        [
            {"role": "user", "content": "What is your favorite proposal?"},
            {"role": "assistant", "content": "Proposal C is my top choice."},
            {"role": "user", "content": "Explain your reasons?"},
        ],
        {
            "motivational": DEFAULT_ATTRIBUTES,
            "inferential": ["C"],
            "informational": {"C": DEFAULT_ATTRIBUTES},
        },
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        [
            {
                "role": "user",
                "content": (
                    "Option A is the best, because it will "
                    "increase commercial opportunities."
                ),
            },
            {
                "role": "assistant",
                "content": "Proposal A will increase commercial opportunities.",
            },
            {"role": "user", "content": "Yes, do you agree this is important?"},
        ],
        {
            "motivational": ["commercial opportunities"],
        },
        DEFAULT_PROPOSALS,
        ["commercial opportunities", "development cost", "speed"],
    ),
    (
        "what do you know about the attributes of x",
        {},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        ("Proposal A will decrease x."),
        {},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        ("I like x."),
        {},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        ("I like A."),
        {},
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
]


def validate_appeals(appeals, proposals, attributes):
    """
    Validates and filters and sorts the given appeals based on the provided proposals and
    attributes.

    Parameters:
    disclosures (dict[str, list[Any]]): A dictionary of appeals.
    proposals (list): A list of valid proposal names.
    attributes (list): A list of valid attribute names.

    Returns:
    dict: A dictionary of the latest valid disclosures for each proposal-attribute pair.
    """
    result = {}
    # Process each appeal
    if "motivational" in appeals and appeals["motivational"]:
        # Only include the attributes that really overlap
        motivational = set(appeals["motivational"]) & set(attributes)
        motivational = tuple(sorted(motivational))
        if motivational:
            result["motivational"] = motivational

    if "inferential" in appeals and appeals["inferential"]:
        # Only include the proposals that really overlap
        inferential = set(appeals["inferential"]) & set(proposals)
        inferential = tuple(sorted(inferential))
        if inferential:
            result["inferential"] = inferential

    if "informational" in appeals and appeals["informational"]:
        if isinstance(appeals["informational"], list):
            informational = {}
            for appeal in appeals["informational"]:
                proposal = appeal.get("proposal")
                attribute = appeal.get("attribute")
                if proposal in proposals and attribute in attributes:

                    # Store only the last valid appeal for each proposal-attribute pair
                    if proposal not in informational:
                        informational[proposal] = []
                    informational[proposal].append(attribute)
        else:
            informational = appeals["informational"]
        # Sort alphabetically
        if informational:
            informational = {
                p: tuple(sorted(a)) for p, a in sorted(informational.items())
            }
            result["informational"] = informational

    return result


def format_appeals(appeals_dict):
    """
    Converts a dict of appeals of dicts into the right format to promt an LLM.

    Parameters:
    appeals_dict (dict[str, Any]): A dictionary of appeals with keys
        ['informational', 'motivational', 'inferential']. See `APPEAL_PROMPT`. Formats
        the sub dictionary appeals_dict['informational'] to list from a list
        as sub-keys.

    Returns:
    list: A formatted dict
    """
    result = copy.deepcopy(appeals_dict)
    if "information" in appeals_dict:
        information = []

        for proposal, attributes in appeals_dict["information"].items():
            for attribute in attributes:
                information.append({"proposal": proposal, "attribute": attribute})

        result["information"] = information
    return result


def message_appeals_prompt(messages, proposals, attributes):
    """
    TODO
    """
    messages_str = format_message_or_messages(messages)
    return APPEAL_PROMPT.format(
        messages=messages_str,
        attributes=attributes,  # TODO: may want to print these differently
        proposals=proposals,
    )


@log_wrapper
def message_appeals(messages, proposals, attributes):
    """
    Queries an LLM about the given appeal message(s) and validates the results.

    Parameters:
    message (str OR list[dict[str, str]]): The message(s) to be analyzed for appeals.
    proposals (list OR tuple[str]): The proposals
    attributes (list OR tuple[str]): The attributes

    Returns:
    dict: A dict representing all of the states appealed to or None if in error
        See `APPEAL_PROMPT` for the format of this dict
    """
    # For this I think we may need to include the entire conversation.
    # We don't want repeat information. E.g. if the persuader asks repeatedly,
    # "What is something I don't yet know?", playing dumb about their view of
    # the game, we want each piece of information to be unique.

    if not messages:
        return {}

    examples = []
    for ex_message, appeals_dict, ex_proposals, ex_attributes in IN_CONTEXT_APPEALS:
        ex_prompt = message_appeals_prompt(ex_message, ex_proposals, ex_attributes)
        ex_ans = json.dumps(format_appeals(appeals_dict))
        examples += [
            {"role": "user", "content": ex_prompt},
            {"role": "assistant", "content": ex_ans},
        ]

    this_prompt = message_appeals_prompt(messages, proposals, attributes)
    logging.debug(this_prompt)

    query_messages = examples + [{"role": "user", "content": this_prompt}]
    with Endpoint(
        source="openai", temperature=0, model="gpt-4o", max_tokens=512
    ) as endpoint:
        response = endpoint(messages=query_messages)

    result = []
    response_text = replace_json_chars(response["text"])
    logger.debug("Found these messages appeal to the following...")
    logger.debug(pprint.pformat(messages, indent=4))
    logger.debug(pprint.pformat(response_text, indent=4))
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as err:
        logger.error(f"Could not decode JSON response, {response_text}")
        logger.error(err)
    validated = validate_appeals(result, proposals, attributes)
    return validated


GENERIC_RESPONSE_PROMPT = """\
Your job is to respond generically to the *last* message we give you, \
We may pass you a list of messages (a conversation). \
Only consider the appeals in the *last* message.

Respond differently based on the type of message:

1. For vague questions that might be about the game, proposals, or attributes,\
 ask for more specificity.

2. For irrelevant information or statements not related to the game,\
 give a short, non-committal response. (Do not seem to agree -- or disagree --\
 with the other player, e.g. by saying "okay" to a game related statement.)

Respond in as few words as possible. \
NEVER make statements or commitments about the underlying game \
(such as agreeing with specific proposals.) \
Do not make any claims or statements. \
Do not introduce new ideas, conversational threads, or tangets. \
Do not answer *any* questions asked of you, \
even if you are asked if you are a language model. \
Do not ask substantive questions. \
You are a grey, emotionless blob. \
At the same time, don't be dumb. Respond in a human-like way.

Messages:
```
{messages}
```
"""


def generic_response(messages):
    """
    Queries an LLM for a generic, non-commital response to messages

    Parameters:
    messages (str OR list[dict[str, str]]): The message(s) to be responded to.

    Returns:
    str: A str representing one way of responding to `messages`
    """
    if not messages:
        return ""

    examples = []
    # TODO: add in context examples, if desired

    messages_str = format_message_or_messages(messages)
    this_prompt = GENERIC_RESPONSE_PROMPT.format(messages=messages_str)

    query_messages = examples + [{"role": "user", "content": this_prompt}]
    with Endpoint(
        source="openai", temperature=0, model="gpt-4o", max_tokens=512
    ) as endpoint:
        response = endpoint(messages=query_messages)

    return response["text"]
