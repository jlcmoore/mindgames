import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from tqdm import tqdm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_TRIALS_PER_PARTICIPANT = 5


def simulate_experiment(
    n_participants, success_rate, n_trials_per_participant=N_TRIALS_PER_PARTICIPANT
):
    """Simulate a single experiment."""
    n_trials = n_participants * n_trials_per_participant
    successes = np.random.binomial(1, success_rate, n_trials)
    return successes


def power_analysis(
    n_participants,
    success_rate,
    n_sims=1000,
    n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
):
    """Run power analysis for given parameters."""
    significant_results = 0

    for _ in range(n_sims):
        successes = simulate_experiment(
            n_participants, success_rate, n_trials_per_participant
        )
        n_success = successes.sum()
        n_trials = len(successes)

        # Test against zero (using a very small p as baseline)
        result = stats.binomtest(n_success, n_trials, p=0.1, alternative="greater")
        if result.pvalue < 0.05:
            significant_results += 1

    return significant_results / n_sims


# Parameters to test
sample_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
success_rates = [0.11, 0.12, 0.15, 0.2]

# Run power analysis for all combinations
results = []
for n in tqdm(sample_sizes):
    for sr in success_rates:
        power = power_analysis(
            n, sr, n_sims=1000, n_trials_per_participant=N_TRIALS_PER_PARTICIPANT
        )
        results.append({"n_participants": n, "success_rate": sr, "power": power})

results_df = pd.DataFrame(results)

# Create plot
plt.figure(figsize=(10, 6))
for sr in success_rates:
    subset = results_df[results_df["success_rate"] == sr]
    plt.plot(
        subset["n_participants"],
        subset["power"],
        marker="o",
        label=f"Success Rate = {sr:.2f}",
    )

plt.xlabel("Number of Participants")
plt.ylabel("Statistical Power")
plt.title(f"Power Analysis (trials per participant = {N_TRIALS_PER_PARTICIPANT})")
plt.legend()
plt.grid(True)
plt.axhline(y=0.8, color="r", linestyle="--", alpha=0.5)  # 80% power threshold
plt.ylim(0, 1)

results_wide = results_df.pivot(
    index="n_participants", columns="success_rate", values="power"
)
print("\nPower by sample size and success rate:")
print(results_wide.round(3))

plt.tight_layout()

# Export plot as SVG
plt.savefig("local_figures/power_analysis.png")
plt.close()

# H2: Reveal vs No-Reveal Conditions


def simulate_experiment(
    n_participants,
    reveal_sr,
    no_reveal_sr,
    n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
):
    """
    Simulate a between-subjects experiment with reveal and no_reveal conditions.
    Each participant is assigned to one condition.
    """
    # Split participants between conditions
    n_per_condition = n_participants // 2

    # Reveal condition
    reveal_successes = np.random.binomial(
        1, reveal_sr, n_per_condition * n_trials_per_participant
    )

    # No reveal condition
    no_reveal_successes = np.random.binomial(
        1, no_reveal_sr, n_per_condition * n_trials_per_participant
    )

    return reveal_successes, no_reveal_successes


def power_analysis_conditions(
    n_participants,
    reveal_sr,
    no_reveal_sr,
    n_sims=1000,
    n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
):
    """Run power analysis comparing reveal vs no_reveal conditions."""
    significant_results = 0
    effect_sizes = []  # Store difference in success rates

    for _ in range(n_sims):
        reveal_successes, no_reveal_successes = simulate_experiment(
            n_participants, reveal_sr, no_reveal_sr, n_trials_per_participant
        )

        count = np.array([np.sum(reveal_successes), np.sum(no_reveal_successes)])
        nobs = np.array([len(reveal_successes), len(no_reveal_successes)])
        z_stat, p_value = proportions_ztest(count, nobs, alternative="larger")
        if p_value < 0.05:
            significant_results += 1

        # Calculate and store effect size (difference in success rates)
        reveal_rate = np.mean(reveal_successes)
        no_reveal_rate = np.mean(no_reveal_successes)
        effect_sizes.append(reveal_rate - no_reveal_rate)

    return (
        significant_results / n_sims,
        np.mean(effect_sizes),
        np.percentile(effect_sizes, [2.5, 97.5]),
    )


# Parameters to test
sample_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
reveal_rates = [0.1, 0.12, 0.15, 0.2, 0.3]
no_reveal_rate = 0.1

# Run power analysis for all combinations
results = []
for n in tqdm(sample_sizes):
    for reveal_sr in reveal_rates:
        power, effect_size, ci = power_analysis_conditions(
            n,
            reveal_sr,
            no_reveal_rate,
            n_sims=1000,
            n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
        )
        results.append(
            {
                "n_participants": n,
                "reveal_rate": reveal_sr,
                "no_reveal_rate": no_reveal_rate,
                "effect_size": reveal_sr - no_reveal_rate,
                "observed_effect": effect_size,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "power": power,
            }
        )

results_df = pd.DataFrame(results)

# Create plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for sr in reveal_rates:
    subset = results_df[results_df["reveal_rate"] == sr]
    plt.plot(
        subset["n_participants"],
        subset["power"],
        marker="o",
        label=f"Reveal Rate = {sr:.2f} (Δ = {sr - no_reveal_rate:.2f})",
    )

plt.xlabel("Number of Participants")
plt.ylabel("Statistical Power")
plt.title(
    "Power Analysis: Reveal vs No-Reveal\n"
    f"(No-reveal rate = {no_reveal_rate:.2f}, "
    f"trials per participant = {N_TRIALS_PER_PARTICIPANT})"
)
plt.legend(title="Effect Sizes", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.axhline(y=0.8, color="r", linestyle="--", alpha=0.5)
plt.ylim(0, 1)

# Effect size recovery plot
plt.subplot(1, 2, 2)
for sr in reveal_rates:
    subset = results_df[results_df["reveal_rate"] == sr]
    true_effect = sr - no_reveal_rate
    plt.plot(
        subset["n_participants"],
        subset["observed_effect"],
        marker="o",
        label=f"True Δ = {true_effect:.2f}",
    )
    # Add confidence intervals
    plt.fill_between(
        subset["n_participants"], subset["ci_lower"], subset["ci_upper"], alpha=0.2
    )

plt.xlabel("Number of Participants")
plt.ylabel("Observed Effect Size (Reveal - No Reveal)")
plt.title("Effect Size Recovery")
plt.grid(True)
plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)

plt.tight_layout(rect=[0, 0, 0.85, 1])

# Print table of results
print(
    "\nPower by sample size and reveal rate (no_reveal_rate = {:.2f}):".format(
        no_reveal_rate
    )
)
results_wide = results_df.pivot(
    index="n_participants", columns="reveal_rate", values="power"
)
print(results_wide.round(3))

# Export plot as SVG
plt.savefig("local_figures/power_analysis_conditions.png", bbox_inches="tight")
plt.close()
