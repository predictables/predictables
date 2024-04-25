from __future__ import annotations
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 100000
SD_ADJUSTMENT_FACTOR = 10


# Adjusted probability calculation function with enhanced noise
def calculate_probabilities(base_prob: float, levels: int = 4) -> float:
    if levels <= 1:
        raise ValueError(
            "The number of levels must be greater than one to generate noise properly."
        )

    logits = np.log(
        max(min(base_prob, 1 - 1e-10), 1e-10)
        / (1 - max(min(base_prob, 1 - 1e-10), 1e-10))
    )
    sd = np.sqrt(np.abs(logits) * SD_ADJUSTMENT_FACTOR)
    noises = np.random.default_rng(RANDOM_SEED).normal(0, sd, levels - 1)
    last_noise = -np.sum(noises)
    all_noises = np.append(noises, last_noise)
    adjusted_logits = logits + all_noises
    probabilities = 1 / (1 + np.exp(-adjusted_logits))

    return probabilities[0]  # type: ignore


# Function to generate the full dataset with probabilities
def generate_full_naics_data() -> pd.DataFrame:
    base_probs = {
        f"{10+i}": p
        for i, p in enumerate(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        )
    }
    data = []
    for naics_2, base_prob in base_probs.items():
        for j in range(1, 6):
            naics_3 = f"{naics_2}{j}"
            for k in range(1, 5):
                naics_4 = f"{naics_3}{k}"
                for l in range(1, 4):  # noqa: E741
                    naics_5 = f"{naics_4}{l}"
                    for m in range(1, 4):
                        naics_6 = f"{naics_5}{m}"
                        probability = calculate_probabilities(base_prob, 4)
                        data.append(
                            {
                                "naics_2_cd": naics_2,
                                "naics_3_cd": naics_3,
                                "naics_4_cd": naics_4,
                                "naics_5_cd": naics_5,
                                "naics_6_cd": naics_6,
                                "probability": probability,
                            }
                        )
    return pd.DataFrame(data)


def sample_from_hierarchy(data: pd.DataFrame) -> pd.DataFrame:
    return data.sample(
        n=N_SAMPLES, replace=True, weights="probability", random_state=RANDOM_SEED
    )


def randomly_draw_target(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        target=data["probability"].apply(
            lambda p: np.random.default_rng(RANDOM_SEED).binomial(1, p)
        )
    )


# Function to sample from the generated data and assign targets
def sample_and_set_targets(data: pd.DataFrame) -> pd.DataFrame:
    return randomly_draw_target(sample_from_hierarchy(data))


# Validation functions
def validate_data(data: pd.DataFrame) -> tuple[bool, pd.Series, float]:
    # Check hierarchical structure
    structure_check = True
    for row in data.iterrows():
        if not all(
            [
                row[1]["naics_3_cd"].startswith(row[1]["naics_2_cd"]),
                row[1]["naics_4_cd"].startswith(row[1]["naics_3_cd"]),
                row[1]["naics_5_cd"].startswith(row[1]["naics_4_cd"]),
                row[1]["naics_6_cd"].startswith(row[1]["naics_5_cd"]),
            ]
        ):
            structure_check = False
            break

    # Probability and target consistency
    probability_stats = data["probability"].describe()
    target_stats = data["target"].mean()

    return structure_check, probability_stats, target_stats


# Main execution block
if __name__ == "__main__":
    full_data = generate_full_naics_data()
    sampled_data = sample_from_hierarchy(full_data)
    sampled_data_with_targets = randomly_draw_target(sampled_data)

    # Perform validation
    structure_valid, prob_stats, mean_target = validate_data(sampled_data_with_targets)
    print("Data Generation and Validation Complete")  # noqa: T201
    print(f"Hierarchical Structure Valid: {structure_valid}")  # noqa: T201
    print(f"Probability Stats:\n{prob_stats}")  # noqa: T201
    print(f"Mean Target: {mean_target}")  # noqa: T201

    # Export to Parquet
    sampled_data_with_targets.to_parquet("final_naics_data.parquet")
