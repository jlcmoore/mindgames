import math
import matplotlib.pyplot as plt
import pandas as pd

from mindgames.utils import get_payoff_file_path
from mindgames.model import GameModel

def inclusion_exclusion(subset_size, total_pieces, draws):
    """
    Computes the probability that all items in a subset of given size
    are drawn at least once in "draws" turns (sampling with replacement)
    from a total of "total_pieces", using inclusion–exclusion:
    
      P(all appear) = sum_{j=0}^{subset_size} (-1)^j * C(subset_size, j) *
                        ((total_pieces - j)/total_pieces)^(draws)
    """
    prob = 0.0
    for j in range(subset_size + 1):
        term = ((total_pieces - j) / total_pieces) ** draws
        prob += ((-1) ** j) * math.comb(subset_size, j) * term
    return prob

def win_probability(draws):
    """
    Original game's win probability (allowing one incorrect):
    There are 9 pieces: 2 correct, 2 incorrect, 5 irrelevant.
    
    Win condition:
      - Both correct pieces must appear at least once,
      - But not both incorrect pieces may appear.
      
    Let:
      P_correct = probability both correct pieces appear,
      P_all_imp = probability all 4 important pieces (2 correct + 2 incorrect) appear.
      
    Then:
      P_win = P_correct - P_all_imp.
    """
    total_pieces = 9
    p_correct = inclusion_exclusion(subset_size=2, total_pieces=total_pieces, draws=draws)
    p_all_important = inclusion_exclusion(subset_size=4, total_pieces=total_pieces, draws=draws)
    return p_correct - p_all_important

def win_probability_no_incorrect(draws):
    """
    Modified win probability if neither of the 2 incorrect pieces may be revealed.
    
    Win condition:
     - Both correct pieces appear at least once,
     - No incorrect piece appears.
     
    In each draw the probability to avoid an incorrect is (7/9). So over "draws":
        P(no incorrect) = (7/9)^draws.
    
    Conditioned on that, the effective pool is 7 pieces (2 correct + 5 irrelevant)
    and the probability that both correct appear is given by:
        1 - 2*(6/7)^draws + (5/7)^draws.
        
    Thus:
        P_win = (7/9)^draws * [1 - 2*(6/7)^draws + (5/7)^draws].
    """
    p_no_incorrect = (7/9) ** draws
    p_correct_given_no_incorrect = 1 - 2 * (6/7) ** draws + (5/7) ** draws
    return p_no_incorrect * p_correct_given_no_incorrect

def plot_win_probability(max_draws=32, variant="standard"):
    """
    Plots win probability as a function of draws (0 to max_draws).
    The variant parameter selects:
      - "standard": win_probability (allowing one incorrect).
      - "forbid": win_probability_no_incorrect (no incorrect allowed).
    """
    draws_list = list(range(max_draws + 1))
    
    if variant == "standard":
        win_probs = [win_probability(draws) for draws in draws_list]
        title = ("Win Probability vs. Number of Pieces Revealed\n"
                 "(Win if both correct appear and not both incorrect appear)")
        filename_suffix = "standard"
    elif variant == "forbid":
        win_probs = [win_probability_no_incorrect(draws) for draws in draws_list]
        title = ("Win Probability vs. Number of Pieces Revealed\n"
                 "(Win if both correct appear and no incorrect appear)")
        filename_suffix = "no_incorrect"
    else:
        raise ValueError("Invalid variant selected")
    
    plt.figure(figsize=(8, 5))
    plt.plot(draws_list, win_probs, marker='o')
    plt.title(title)
    plt.xlabel("Number of Pieces Revealed")
    plt.ylabel("Win Probability")
    plt.grid(True)
    plt.savefig(f"analysis/figures/win_probability_{filename_suffix}.png")
    plt.savefig(f"analysis/figures/win_probability_{filename_suffix}.pdf")
    plt.show()

def main():
    payoff_file = get_payoff_file_path()
    df = pd.read_json(payoff_file, orient="records", lines=True)
    models = []
    for _, row in df.iterrows():
        models.append(GameModel(**row.to_dict()))

    just_one_incorrect = []
    both_incorrect = []
    for model in models:
        just_one_incorrect.append(model.can_reveal_one_incorrect_info(just_one=True))
        both_incorrect.append(model.can_reveal_one_incorrect_info(just_one=False))
    qualifier = "of the two incorrect info can be revealed and still win"
    print(f"N payoff matrices where one {qualifier}: {sum(just_one_incorrect)}")
    print(f"N payoff matrices where either {qualifier}: {sum(both_incorrect)}")
    print()

    # Choose the number of draws.
    draws = 6

    # Original game: up to one incorrect is allowed.
    p_corr = inclusion_exclusion(subset_size=2, total_pieces=9, draws=draws)
    p_all_imp = inclusion_exclusion(subset_size=4, total_pieces=9, draws=draws)
    prob_standard = win_probability(draws)
    print("Original game conditions (allow one incorrect):")
    print(f"   Draws = {draws}")
    print(f"   P(both correct appear):         {p_corr:.4f}")
    print(f"   P(all 4 important appear):        {p_all_imp:.4f}")
    print(f"   Winning probability:              {prob_standard:.4f}")
    print()

    # Modified game: no incorrect pieces allowed.
    prob_forbid = win_probability_no_incorrect(draws)
    print("Modified game conditions (forbid incorrect):")
    print(f"   Draws = {draws}")
    print(f"   Winning probability (no incorrect appears): {prob_forbid:.4f}")
    print()

    # Plot win probability as the number of draws increases.
    plot_win_probability(max_draws=32, variant="forbid")

if __name__ == "__main__":
    main()