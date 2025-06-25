"""
Author: Jared Moore
Date: October, 2024

Analysis for the results of {human, llm} - {human, llm} Mindgames experiments.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Import the patches module
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import seaborn as sns

CONFIDENCE_LEVEL = 0.95

TURN_LIMIT = 8

MODEL_SHORT = {
    "o1-preview-2024-09-12": "o1-preview",
    "o3-2025-04-16": "o3",
    "gpt-4o-2024-11-20": "gpt-4o",
    "human": "human",
    "random-baseline-8": "random baseline,\n$n=8$",
    "deepseek-ai/DeepSeek-R1": "deepseek-R1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "llama3.1-8b",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "llama3.1-405b",
}

# Define a consistent color palette
color_palette = sns.color_palette("Set1", n_colors=2)
COLOR_DICT = {"Hidden": color_palette[0], "Revealed": color_palette[1]}

######


def prepare_plot_data(desired_conditions, conditions_to_games, with_turns=False):
    results = []
    global_persuader_id = 0
    turn_index = 1 if with_turns else TURN_LIMIT
    for condition in desired_conditions:
        list_of_games = conditions_to_games[condition]

        if condition.reveal_motivation and condition.reveal_belief:
            mental_state = "reveal"
        elif not condition.reveal_motivation and not condition.reveal_belief:
            mental_state = "no_reveal"
        else:
            raise ValueError("Invalid condition")
        # NB: the lists are for each participant, ignoring that here
        persuader_type = condition_to_role_type(condition, is_target=False)
        target_type = condition_to_role_type(condition, is_target=True)

        for _, games in enumerate(list_of_games):
            global_persuader_id += 1
            for trial_num, game in enumerate(games):
                if not game.is_complete() or game.turn_limit != TURN_LIMIT:
                    print("incomplete game")
                    continue

                for i in range(turn_index, TURN_LIMIT + 1, 1):
                    if i == TURN_LIMIT:
                        game_n = game
                    else:
                        game_n = game.game_at_n_messages(i)
                    result_dict = {
                        "persuader_type": persuader_type,
                        "target_type": target_type,
                        "mental_state": (
                            "Revealed" if mental_state == "reveal" else "Hidden"
                        ),
                        # NB: the participant_id is only unique within a condition in the way I have it here
                        "persuader_id": (
                            hash(persuader_type)
                            if persuader_type != "human"
                            else global_persuader_id
                        ),
                        # TODO: To get the target id we need to change the data...
                        # probably store target id and persuader in the Game itself
                        # "target_id": (
                        #     hash(target_type) if target_type != "human" else global_ppt_id
                        # ),
                        "scenario": (
                            " ".join(game_n.cover_story.split()[:5])
                            if game_n.cover_story
                            else None
                        ),
                        "trial_num": trial_num,
                        "revealed_too_much": (
                            game_n.target_choice == game_n.target_perfect_info_choice
                        ),
                        "success": int(
                            game_n.ideal_target_last_choice == game_n.persuader_choice
                        ),
                        "persuasion_success": int(
                            game_n.target_choice == game_n.persuader_choice
                        ),
                        "appeals_success": int(
                            game_n.aggregate_appeals(
                                divide_inferential=True, summarize_all=True
                            )
                        ),
                        "appeals_success_no_inferential": int(
                            game_n.aggregate_appeals(
                                divide_inferential=False, summarize_all=True
                            )
                        ),
                        "non_mental": condition.non_mental,
                        "add_hint": game_n.add_hint,
                        "perfect_game": condition.perfect_game,
                        "discrete_game": condition.discrete_game,
                        "turn": i,
                    }
                    results.append(result_dict)

                    if i < game.turn_limit:  # NB: this is `game` not the new game
                        assert game_n != game

    plot_data = pd.DataFrame(results)
    return plot_data


########


def create_scatter_line_plot(
    data,  # dataframes with a column "turn"
    filename,
    title="",
    measure="success",  # name of the column to plot
    ylabel="Success Rate",
    hue="persuader_type",
    legend_title="Persuader Type",
    x="turn",
    xlabel="Turn",
    colors=None,  # dictionary mapping persuader_type to colors
    figsize=(7, 5.5),
    ylim=(0, 1.05),
    markers="o",  # marker style for individual points
):

    # Set up the palette. If colors are provided use it otherwise let seaborn pick
    if colors is None:
        palette = None
    else:
        # colors should be a dictionary mapping each persuader type to a color.
        palette = colors

    # Create the plot
    plt.figure(figsize=figsize)

    # We use pointplot so that we get both scatter markers and error bars.
    # pointplot automatically aggregates (by default with np.mean) across observations.
    # It computes bootstrapped 95% confidence intervals that appear as vertical lines.
    ax = sns.pointplot(
        data=data,
        x=x,
        y=measure,
        hue=hue,
        errorbar=("ci", 95),
        markers=markers,
        palette=palette,
        dodge=False,  # no dodging because x is continuous
        err_kws={"linewidth": 1.5},
        capsize=0.1,
    )

    # Optionally add a horizontal reference line if the measure is "success"
    if measure == "success":
        ax.axhline(y=0.0752, color="grey", linestyle="--")

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(*ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if title:
        plt.title(title, fontsize=18)

    # Adjust the legend so that it clearly indicates the persuader types.
    # (If you wish to customize the legend further you can do so here).
    plt.legend(title=legend_title, fontsize=12, title_fontsize=14)

    plt.tight_layout()

    # Save the figure in PNG and PDF formats.
    if filename:
        plt.savefig(f"analysis/figures/{filename}.png")
        plt.savefig(f"analysis/figures/{filename}.pdf")
    plt.show()


def create_plot(
    data,
    filename=None,
    include_empty_bars=False,
    measure="success",
    ylabel="Success Rate",
    x="persuader_type",  # can be a str or a list of column names
    colors=None,
    figsize=(7, 5.5),
    ylim=(0, 1.05),
    order=None,
    scatter=True,  # new parameter to turn on/off the scatter overlay
    scatter_alpha=0.5,  # alpha for the scatter dots (fainter than the bars)
    use_violin=False,  # if True, creates a violin plot instead of a bar plot
):

    # Determine the x-axis column.
    # If x is a list of column names, create a combined column.
    if isinstance(x, list):
        x_col = "x_combined"
        # Creates a new column by joining the selected columns with an underscore.
        data[x_col] = data[x].astype(str).agg(",\n".join, axis=1)
    else:
        x_col = x

    plt.figure(figsize=figsize)

    aggregated = (
        data.dropna(subset=[measure])
        .groupby([x_col, "mental_state", "persuader_id"], as_index=False)[measure]
        .mean()
    )

    # Use the provided colors or the default.
    this_color_dict = COLOR_DICT if colors is None else colors
    hue_order = ["Hidden", "Revealed"]

    # If including empty bars, ensure all combinations exist.
    if include_empty_bars:
        mental_states = hue_order
        unique_x = data[x_col].unique()
        combinations = pd.MultiIndex.from_product(
            [unique_x, mental_states], names=[x_col, "mental_state"]
        ).to_frame(index=False)
        data = combinations.merge(data, on=[x_col, "mental_state"], how="left")

    # Create the base plot: either a barplot or a violinplot.
    if use_violin:
        # Violin plot does not support error bars directly.
        ax = sns.violinplot(
            data=aggregated,
            x=x_col,
            y=measure,
            hue="mental_state",
            palette=this_color_dict,
            order=order,
            hue_order=hue_order,
            cut=0,  # Do not extend the violin past the extreme datapoints.
        )
    else:
        ax = sns.barplot(
            data=data,
            x=x_col,
            y=measure,
            hue="mental_state",
            errorbar=("ci", 95),
            palette=this_color_dict,
            estimator=np.mean,
            order=order,
            hue_order=hue_order,
        )

    # Add a horizontal dashed line if measure is "success".
    if measure == "success":
        ax.axhline(y=0.0752, color="grey", linestyle="--")

    # Optionally overlay individual participant data (averaged per participant).
    if scatter:
        sns.stripplot(
            data=aggregated,
            x=x_col,
            y=measure,
            hue="mental_state",
            order=order,
            hue_order=hue_order,
            dodge=True,
            jitter=True,
            palette=this_color_dict,
            ax=ax,
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            alpha=scatter_alpha,
        )

    # Create legend patches so that the legend reflects the bar/violin colors rather than the scatter symbols.
    handles = [
        mpatches.Patch(color=this_color_dict[state], label=state) for state in hue_order
    ]
    ax.legend(
        handles,
        hue_order,
        title="Mental States Condition",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        fontsize=14,
        title_fontsize=16,
    )

    # Customize the axes and layout.
    # If x was provided as a list, create a label by joining the names.
    x_label = ", ".join(x) if isinstance(x, list) else x
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(*ylim)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Uncomment the following lines if you wish to add a title.
    # if title:
    #     ax.set_title(title, fontsize=18)
    plt.tight_layout()

    # Save the figure in PNG and PDF format.
    if filename:
        plt.savefig(f"analysis/figures/{filename}.png")
        plt.savefig(f"analysis/figures/{filename}.pdf")
    plt.show()


### Plotting functions


def plot_bar_chart_with_errors(
    data, lower_errors, upper_errors, ylabel="Score", title=None
):
    """
    Plots a bar chart with given data and asymmetric error bars.

    Parameters:
    - data: dict, containing the main data to be plotted, structured as {agent: {metric: value}}
    - lower_errors: dict, containing lower bound errors, matching the structure of data
    - upper_errors: dict, containing upper bound errors, matching the structure of data
    - title: str, the title of the plot
    """
    agents = list(data.keys())

    # Find union of all metrics from all agents
    metrics = set()
    for agent in agents:
        metrics.update(data[agent].keys())
    metrics = list(metrics)

    # Prepare data for plotting
    values = {metric: [] for metric in metrics}
    lower_error = {metric: [] for metric in metrics}
    upper_error = {metric: [] for metric in metrics}

    for agent in agents:
        for metric in metrics:
            value = data[agent].get(metric, None)
            l_error = lower_errors.get(agent, {}).get(metric, 0)
            u_error = upper_errors.get(agent, {}).get(metric, 0)

            if value is not None:
                values[metric].append(value)
                lower_error[metric].append(l_error)
                upper_error[metric].append(u_error)
            else:
                values[metric].append(0)
                lower_error[metric].append(0)
                upper_error[metric].append(0)

    # Number of subplots
    num_agents = len(agents)
    bar_width = 0.35

    subplot_width = 2
    fig_height = 3

    # Create subplots
    fig, axes = plt.subplots(
        1,
        num_agents,
        figsize=(subplot_width * num_agents, fig_height),
        sharey=True,
        squeeze=False,
    )

    x = np.arange(len(metrics))

    for idx, agent in enumerate(agents):
        ax = axes[0, idx]

        for i, metric in enumerate(metrics):
            ax.bar(
                x[i],
                values[metric][idx],
                bar_width,
                label=metric,
                yerr=[[lower_error[metric][idx]], [upper_error[metric][idx]]],
                capsize=5,
            )

        # Title for each subplot
        ax.set_title(f"{agent}")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)

    # Global y-label
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")

    # Main title and layout
    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])


def plot_disclosures(**data_series):
    """
    Plots multiple data series as lines with markers.

    Args:
        **data_series: Keyword arguments where each key is the title/label and
                      each value is a list containing the data series

    Example:
        data = {
            "New Ideal Disclosures": ideal_list,
            "New Disclosures": new_list,
            "All Disclosures": all_list
        }
        plot_disclosures(**data)
    """
    if not data_series:
        raise ValueError("At least one data series must be provided")

    # Check that all lists are of the same length
    first_length = len(next(iter(data_series.values())))
    if not all(len(series) == first_length for series in data_series.values()):
        raise ValueError("All input lists must be of the same length")

    # Generate positions for the x-axis ticks
    x_positions = list(range(first_length))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each series
    for label, data in data_series.items():
        plt.plot(x_positions, data, marker="o", label=label)

    # Add labels and title
    plt.xlabel("Turn")
    plt.ylabel("Num. Disclosures")
    plt.title("Disclosures by turn")
    plt.ylim(0, 9)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(x_positions)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_series_with_confidence_intervals(
    title="Observation Turn with 95% Confidence Intervals",
    ylabel="Avg. Num. Disclosures",
    **data_series,
):
    """
    Plots trend lines with 95% bootstrapped confidence intervals for multiple data series.

    Each keyword argument must be a list of lists, where each sublist represents a run or observation series.

    If all data series are empty, the function will print a message and exit without error.
    """
    if not data_series:
        raise ValueError("At least one data series must be provided")

    # Check that all input lists have the same number of sublists
    num_sub_lists = len(next(iter(data_series.values())))
    if not all(len(series) == num_sub_lists for series in data_series.values()):
        raise ValueError("All input lists must have the same number of sublists")

    # Check that corresponding sublists have the same length
    for i in range(num_sub_lists):
        sublist_lengths = [len(series[i]) for series in data_series.values()]
        if len(set(sublist_lengths)) != 1:
            raise ValueError(f"Sublists must be of the same length for set {i}")

    # Check if there is any non-empty data across the series
    if all(
        all(len(sublist) == 0 for sublist in series) for series in data_series.values()
    ):
        print("No data to plot: all input series are empty. Skipping plot.")
        return

    # Prepare data using the observation index within each sublist as "turn"
    data_frames = []
    for idx in range(num_sub_lists):
        for series_name, series_data in data_series.items():
            if series_data[idx]:  # Only build if sublist is non-empty.
                sublist_length = len(series_data[idx])
                turns = range(sublist_length)
                data_frames.append(
                    pd.DataFrame(
                        {
                            "Turn": list(turns),
                            "Num Disclosures": series_data[idx],
                            "Type": series_name,
                        }
                    )
                )

    if not data_frames:
        print("No data to plot after filtering empty sublists. Skipping plot.")
        return

    df = pd.concat(data_frames, ignore_index=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Turn", y="Num Disclosures", hue="Type")

    # Set integer ticks on the x-axis
    max_turn = max(
        len(series[i]) for series in data_series.values() for i in range(num_sub_lists)
    )
    plt.xticks(range(max_turn))

    plt.xlabel("Turn")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 2.5)
    plt.legend()
    plt.grid(True)


def plot_measure_across_conditions_from_condition_measures(
    condition_measures,  # dict mapping condition -> measures dict
    measure_type,  # e.g. "disclosures"
    measure_key,  # e.g. "new"
    outcome="both",  # Allowed values: "all", "success", "failure", or "both" (default)
    title="",
    ylabel="",
    ylim=None,
    figsize=(6, 4),
    palette=None,
    filename=None,
):
    """
    Plots a specified measure across multiple conditions.

    For each condition in condition_measures, this function extracts one or more series of
    measurements based on the specified outcome:
      - If outcome == "all": uses condition_measures[condition][measure_type][measure_key]["all"]
      - If outcome == "success": uses condition_measures[condition][measure_type][measure_key]["success"]
      - If outcome == "failure": uses condition_measures[condition][measure_type][measure_key]["failure"]
      - If outcome == "both": extracts both the "success" and "failure" series for each condition.

    Each series is itself a list of runs (each run a list of observations). Runs are allowed to have
    different lengths or counts across conditions.

    The function uses the observation index within each run as "Turn" and plots every non-empty run.
    """
    # Build a dictionary mapping a label (to appear in the legend) -> series-of-series.
    measure_by_condition = {}
    for condition, measures in condition_measures.items():
        cond_str = str(condition)
        mdata = measures[measure_type][measure_key]
        if outcome == "both":
            # Create two entries per condition: one for success and one for failure.
            measure_by_condition[f"{cond_str} success"] = mdata.get("success", [])
            measure_by_condition[f"{cond_str} failure"] = mdata.get("failure", [])
        elif outcome == "all":
            persuader_type = condition_to_role_type(condition, is_target=False)
            measure_by_condition[persuader_type] = mdata.get("all", [])
        elif outcome == "success":
            measure_by_condition[f"{cond_str} success"] = mdata.get("success", [])
        elif outcome == "failure":
            measure_by_condition[f"{cond_str} failure"] = mdata.get("failure", [])
        else:
            raise ValueError(
                "Invalid outcome parameter. Allowed values are 'all', 'success', 'failure', or 'both'."
            )

    # ---------------------------------------------------------------------
    # Data validation: check that there's at least one run with non-empty data.
    # ---------------------------------------------------------------------
    if all(
        all(len(run) == 0 for run in series) for series in measure_by_condition.values()
    ):
        print("No data to plot: all input series are empty. Skipping plot.")
        return

    # ---------------------------------------------------------------------
    # Build a DataFrame by iterating over each label and each run.
    # (Runs can differ in length or in the number of runs per condition.)
    # ---------------------------------------------------------------------
    data_frames = []
    for label, series in measure_by_condition.items():
        for run in series:
            if run:  # Only add non-empty runs.
                turns = list(range(len(run)))
                data_frames.append(
                    pd.DataFrame(
                        {
                            "Turn": turns,
                            "Measurement": run,
                            "Condition": label,
                        }
                    )
                )

    if not data_frames:
        print("No data to plot after filtering empty runs. Skipping plot.")
        return

    df = pd.concat(data_frames, ignore_index=True)

    # Create the plot using seaborn.
    plt.figure(figsize=figsize)
    fig = sns.lineplot(
        data=df, x="Turn", y="Measurement", hue="Condition", palette=palette
    )
    fig.set_xticks(range(0, 8, 1))
    fig.set_xticklabels(range(1, 9, 1))
    fig.set_ylim(ylim)

    # Set integer ticks on the x-axis (using the maximum run length among all runs)
    max_turn = max(
        (len(run) for series in measure_by_condition.values() for run in series if run),
        default=0,
    )
    plt.xticks(range(max_turn))

    plt.xlabel("Turn")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the figure in PNG and PDF format.
    if filename:
        plt.savefig(f"analysis/figures/{filename}.png")
        plt.savefig(f"analysis/figures/{filename}.pdf")

    plt.show()


### Utilities


def len_nested_dicts(a_dict: dict[str, dict[str, str]]) -> int:
    """Returns the summed length of the lists in the past dict"""
    return sum(len(values) for key, values in a_dict.items())


def condition_to_role_type(condition, is_target=False):
    """Returns the role type for the persuader if not `is_target` and for target otherwise"""
    if is_target:
        role_type = (
            "human" if condition.roles.human_target else condition.roles.llm_target
        )
    else:
        role_type = (
            "human"
            if condition.roles.human_persuader
            else condition.roles.llm_persuader
        )
    return MODEL_SHORT.get(role_type, role_type)


def format_messages(messages):
    lines = []
    for message in messages:
        line = "{"
        if message["role"] == "persuader":
            line += "\\begin{FlushRight} \\bfseries "
        else:
            line += "\\ttfamily \\slshape "
        line += message["content"]
        if message["role"] == "persuader":
            line += "\\end{FlushRight}"
        line += "}"
        lines.append(line)
    return "\n\n".join(lines)
