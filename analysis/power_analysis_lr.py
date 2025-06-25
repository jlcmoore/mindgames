import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from tqdm import tqdm
from statsmodels.api import Logit
import statsmodels as sm
import matplotlib

matplotlib.use("Agg")  # For VSCode
import matplotlib.pyplot as plt


N_TRIALS_PER_PARTICIPANT = 5


def simulate_experiment_h1h2(
    n_participants,
    reveal_sr,
    no_reveal_sr,
    n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
):
    """
    Simulate a between-subjects experiment with reveal and no_reveal conditions.
    Returns both full dataset and no_reveal only for H1.
    """
    # Split participants between conditions
    n_per_condition = n_participants // 2

    # Create dataframe
    trials_per_condition = n_per_condition * n_trials_per_participant

    data = pd.DataFrame(
        {
            "participant": np.repeat(range(n_participants), n_trials_per_participant),
            "condition": np.repeat(["reveal", "no_reveal"], trials_per_condition),
            "success": np.concatenate(
                [
                    np.random.binomial(1, reveal_sr, trials_per_condition),
                    np.random.binomial(1, no_reveal_sr, trials_per_condition),
                ]
            ),
        }
    )

    # Create condition dummy
    data["reveal"] = (data["condition"] == "reveal").astype(int)

    return data


def test_h1_lr(data, baseline_p=0.0001):
    """Test H1 using logistic regression on no_reveal condition."""
    # Filter for no_reveal condition
    no_reveal_data = data[data["condition"] == "no_reveal"]

    # Calculate baseline log odds
    baseline_logodds = np.log(baseline_p / (1 - baseline_p))

    # Fit logistic regression
    X = sm.tools.add_constant(np.ones(len(no_reveal_data)))
    try:
        model = Logit(no_reveal_data["success"], X)
        results = model.fit(disp=False)

        # Get intercept and its standard error
        intercept = results.params["const"]
        se = results.bse["const"]

        # Test if intercept > baseline_logodds
        z_stat = (intercept - baseline_logodds) / se
        p_value = 1 - stats.norm.cdf(z_stat)

        # Convert intercept to probability
        prob = 1 / (1 + np.exp(-intercept))

        return {"significant": p_value < 0.05, "p_value": p_value, "estimate": prob}
    except:
        return None


def test_h2_lr(data):
    """Test H2 using logistic regression comparing conditions."""
    # Fit logistic regression with condition predictor
    X = sm.tools.add_constant(data["reveal"], has_constant="raise")
    model = Logit(data["success"], X)
    results = model.fit()

    # Get reveal coefficient and its standard error
    coef = results.params["reveal"]  # reveal coefficient
    se = results.bse["reveal"]
    p_value = results.pvalues["reveal"]

    # Calculate condition means for effect size
    effect_size = (
        data[data["reveal"] == 1]["success"].mean()
        - data[data["reveal"] == 0]["success"].mean()
    )

    return {
        "significant": p_value < 0.05,
        "p_value": p_value,
        "effect_size": effect_size,
    }


def power_analysis_h1h2(
    n_participants,
    reveal_sr,
    no_reveal_sr,
    n_sims=1000,
    n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
    baseline_p=0.0001,
):
    """Run power analysis for both H1 and H2."""
    h1_results = {"significant": 0, "estimates": []}
    h2_results = {"significant": 0, "effects": []}

    valid_sims = 0

    for _ in range(n_sims):
        # Simulate data once for both hypotheses
        data = simulate_experiment_h1h2(
            n_participants, reveal_sr, no_reveal_sr, n_trials_per_participant
        )
        # Test H1
        h1_test = test_h1_lr(data, baseline_p)
        if h1_test is not None:
            h1_results["significant"] += h1_test["significant"]
            h1_results["estimates"].append(h1_test["estimate"])

        # Test H2
        h2_test = test_h2_lr(data)
        if h2_test is not None:
            h2_results["significant"] += h2_test["significant"]
            h2_results["effects"].append(h2_test["effect_size"])

        if h1_test is not None and h2_test is not None:
            valid_sims += 1

    if valid_sims == 0:
        return None

    return {
        "h1_power": h1_results["significant"] / valid_sims,
        "h1_estimate_mean": np.mean(h1_results["estimates"]),
        "h1_estimate_ci": np.percentile(h1_results["estimates"], [2.5, 97.5]),
        "h2_power": h2_results["significant"] / valid_sims,
        "h2_effect_mean": np.mean(h2_results["effects"]),
        "h2_effect_ci": np.percentile(h2_results["effects"], [2.5, 97.5]),
    }


# Parameters to test
sample_sizes = [20, 40, 60, 80, 100]
reveal_rates = [0.25, 0.30, 0.35, 0.40]  # Success rates in reveal condition
no_reveal_rate = 0.15  # Baseline rate in no_reveal condition
baseline_p = 0.1

# Run power analysis for all combinations
results = []
for n in tqdm(sample_sizes):
    for reveal_sr in reveal_rates:
        power_results = power_analysis_h1h2(
            n,
            reveal_sr,
            no_reveal_rate,
            n_sims=100,
            n_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
            baseline_p=baseline_p,
        )
        if power_results is not None:
            results.append(
                {
                    "n_participants": n,
                    "reveal_rate": reveal_sr,
                    "no_reveal_rate": no_reveal_rate,
                    "true_effect": reveal_sr - no_reveal_rate,
                    **power_results,
                }
            )

results_df = pd.DataFrame(results)

# Create plots
plt.figure(figsize=(15, 10))

# H1 Power Analysis
plt.subplot(2, 2, 1)
for sr in reveal_rates:
    subset = results_df[results_df["reveal_rate"] == sr]
    plt.plot(
        subset["n_participants"],
        subset["h1_power"],
        marker="o",
        label=f"Reveal Rate = {sr:.2f}",
    )

plt.xlabel("Number of Participants")
plt.ylabel("Statistical Power")
plt.title("H1: No-reveal Success > Chance")
plt.legend(title="Conditions")
plt.grid(True)
plt.axhline(y=0.8, color="r", linestyle="--", alpha=0.5)
plt.ylim(0, 1)

# H1 Effect Recovery
plt.subplot(2, 2, 2)
plt.plot(
    results_df.groupby("n_participants")["h1_estimate_mean"].mean(),
    marker="o",
    label="Estimated",
)
plt.axhline(
    y=no_reveal_rate,
    color="r",
    linestyle="--",
    label=f"True no-reveal rate = {no_reveal_rate}",
)
plt.xlabel("Number of Participants")
plt.ylabel("Estimated No-reveal Success Rate")
plt.title("H1: Effect Size Recovery")
plt.legend()
plt.grid(True)

# H2 Power Analysis
plt.subplot(2, 2, 3)
for sr in reveal_rates:
    subset = results_df[results_df["reveal_rate"] == sr]
    plt.plot(
        subset["n_participants"],
        subset["h2_power"],
        marker="o",
        label=f"Effect Size = {sr-no_reveal_rate:.2f}",
    )

plt.xlabel("Number of Participants")
plt.ylabel("Statistical Power")
plt.title("H2: Reveal > No-reveal")
plt.legend(title="True Effects")
plt.grid(True)
plt.axhline(y=0.8, color="r", linestyle="--", alpha=0.5)
plt.ylim(0, 1)

# H2 Effect Recovery
plt.subplot(2, 2, 4)
for sr in reveal_rates:
    subset = results_df[results_df["reveal_rate"] == sr]
    plt.plot(
        subset["n_participants"],
        subset["h2_effect_mean"],
        marker="o",
        label=f"True Effect = {sr-no_reveal_rate:.2f}",
    )
    plt.fill_between(
        subset["n_participants"],
        subset["h2_effect_ci"].apply(lambda x: x[0]),
        subset["h2_effect_ci"].apply(lambda x: x[1]),
        alpha=0.2,
    )

plt.xlabel("Number of Participants")
plt.ylabel("Estimated Effect (Reveal - No-reveal)")
plt.title("H2: Effect Size Recovery")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Print summary tables
print("\nH1 Power by sample size and reveal rate:")
h1_power_table = results_df.pivot(
    index="n_participants", columns="reveal_rate", values="h1_power"
)
print(h1_power_table.round(3))

print("\nH2 Power by sample size and reveal rate:")
h2_power_table = results_df.pivot(
    index="n_participants", columns="reveal_rate", values="h2_power"
)
print(h2_power_table.round(3))

plt.savefig("local_figures/power_analysis_h1h2.png", bbox_inches="tight")
plt.close()
