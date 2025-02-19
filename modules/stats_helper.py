import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os
import matplotlib.patches as mpatches


# Custom JSON encoder to handle numpy data types.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):  # Add this check for Pandas Series
            return obj.tolist()  # Convert Series to a list
        return super(NpEncoder, self).default(obj)


# Function to calculates descriptive statistics for each group in the dataset.
def get_descriptive_stats(dataset):
    # Convert dataset to a Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Group by 'group'
    grouped = df.groupby("group")

    # Initialize results dictionary
    results = {}

    # Define metrics for comprehensive analysis
    metrics = [
        "coverage",
        "clarity",
        "edge_and_negative_cases_score",
        "non_functional_coverage",
        "quality_score",
    ]

    # Define weights for weighted score calculation (sum to 1.0)
    weights = {
        "coverage": 0.30,
        "clarity": 0.20,
        "edge_and_negative_cases_score": 0.25,
        "non_functional_coverage": 0.25,
    }

    for group, group_df in grouped:
        total_tc = len(group_df)
        high_quality_cases = (group_df["quality_score"] >= 4).sum()

        # Weighted score calculation using defined weights
        total_weighted_score = sum(
            (group_df[metric] * weight).sum()  # Sum each metric's weighted scores
            for metric, weight in weights.items()
        )

        # Calculate efficiency index (max possible score per case = 5)
        efficiency_index = total_weighted_score / (5 * total_tc)

        # Initialize group results
        group_results = {
            "total_tc": total_tc,
            "high_quality_cases": high_quality_cases,
            "total_weighted_score": total_weighted_score,
            "qtq_ratio": total_weighted_score / total_tc,
            "efficiency_index": efficiency_index,
        }

        # Calculate comprehensive statistics for all metrics
        for metric in metrics:
            group_results[f"avg_{metric}"] = group_df[metric].mean()
            group_results[f"std_{metric}"] = group_df[metric].std()
            group_results[f"median_{metric}"] = group_df[metric].median()
            mode = group_df[metric].mode()
            group_results[f"mode_{metric}"] = mode.iloc[0] if not mode.empty else None
            group_results[f"var_{metric}"] = group_df[metric].var()

        # Add small sample warning
        if total_tc < 30:
            group_results["warning"] = "Small sample size - results may be unreliable"
        results[group] = group_results

    # convert results into valid JSON
    results_json = json.dumps(results, indent=2, cls=NpEncoder)
    return results_json


# Function to performs statistical tests (ANOVA and pairwise t-tests) on the dataset.
def perform_statistical_tests(dataset, test_metric="coverage"):

    # Convert dataset to a Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Group by 'group'
    grouped = df.groupby("group")

    # Store all group data for ANOVA and t-tests
    all_group_data = {}
    for group, group_df in grouped:
        all_group_data[group] = group_df

    # Perform ANOVA test on specified metric
    anova_groups = [data[test_metric] for data in all_group_data.values()]
    anova_result = f_oneway(*anova_groups)

    # Perform pairwise t-tests with multiple comparison correction
    group_names = list(all_group_data.keys())
    t_test_results = {}
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group1 = group_names[i]
            group2 = group_names[j]
            t_stat, p_value = ttest_ind(
                all_group_data[group1][test_metric], all_group_data[group2][test_metric]
            )
            key = f"{group1}_vs_{group2}"
            t_test_results[key] = {
                "t_stat": t_stat,
                "p_value": p_value,
                "interpretation": (
                    "Significant" if p_value < 0.05 else "Not significant"
                ),
            }

    # Apply Bonferroni correction
    num_comparisons = len(t_test_results)
    for comparison in t_test_results:
        original_p = t_test_results[comparison]["p_value"]
        corrected_p = min(original_p * num_comparisons, 1.0)
        t_test_results[comparison]["corrected_p_value"] = corrected_p

    # Add statistical results to output
    statistical_tests = {
        "anova": {
            "tested_metric": test_metric,
            "F-statistic": anova_result.statistic,
            "p-value": anova_result.pvalue,
        },
        "pairwise_t_tests": t_test_results,
        "note": "Bonferroni correction applied to pairwise comparisons",
    }

    statistical_tests_json = json.dumps(statistical_tests, indent=2, cls=NpEncoder)
    return statistical_tests_json


# Function to convert the "processed_results" scores data into evaluations format
def format_data_to_evaluations(dataset):
    evaluations = []

    for result in dataset:
        evaluations.append(
            {
                # unique id of a test case
                "test_case_id": result["test_case_id"],
                # score per criteria
                "group": result["group"],
                "coverage": result["evaluation"]["coverage"]["score"],
                "clarity": result["evaluation"]["clarity"]["score"],
                "edge_and_negative_cases_score": result["evaluation"][
                    "edge_and_negative_cases_score"
                ]["score"],
                "non_functional_coverage": result["evaluation"][
                    "non_functional_coverage"
                ]["score"],
                # Calculate the overall weighted or quality score
                "quality_score": (
                    0.3 * result["evaluation"]["coverage"]["score"]
                    + 0.2 * result["evaluation"]["clarity"]["score"]
                    + 0.25
                    * result["evaluation"]["edge_and_negative_cases_score"]["score"]
                    + 0.25 * result["evaluation"]["non_functional_coverage"]["score"]
                ),
            }
        )

    return evaluations


# Function to distribute grouped results into performance metrics
def structuring_stats_in_metrics(dataset):
    structured_stats_results = {
        # Key Performance Metrics
        "Key Performance Metrics": {
            "Total Test Cases": {},
            "High Quality Cases": {},
            "Total Weighted Score": {},
            "Quality-to-Quantity Ratio": {},
            "Efficiency Index": {},
        },
        # Individual Metrics
        "Coverage": {"avg": {}, "std": {}, "median": {}, "mode": {}, "var": {}},
        "Clarity": {"avg": {}, "std": {}, "median": {}, "mode": {}, "var": {}},
        "Edge & Negative Cases Score": {
            "avg": {},
            "std": {},
            "median": {},
            "mode": {},
            "var": {},
        },
        "Non-Functional Coverage": {
            "avg": {},
            "std": {},
            "median": {},
            "mode": {},
            "var": {},
        },
        "Overall Quality Score": {
            "avg": {},
            "std": {},
            "median": {},
            "mode": {},
            "var": {},
        },
    }

    # Load the JSON data from the 'dataset' string
    results_data = json.loads(dataset)

    # Now iterate through the loaded dictionary
    for group, stats in results_data.items():
        # grouping all key performance metrics
        structured_stats_results["Key Performance Metrics"]["Total Test Cases"][
            group
        ] = stats["total_tc"]
        structured_stats_results["Key Performance Metrics"]["High Quality Cases"][
            group
        ] = stats["high_quality_cases"]
        structured_stats_results["Key Performance Metrics"]["Total Weighted Score"][
            group
        ] = stats["total_weighted_score"]
        structured_stats_results["Key Performance Metrics"][
            "Quality-to-Quantity Ratio"
        ][group] = stats["qtq_ratio"]
        structured_stats_results["Key Performance Metrics"]["Efficiency Index"][
            group
        ] = stats["efficiency_index"]

        # grouping all individual metrics
        # Coverage
        structured_stats_results["Coverage"]["avg"][group] = stats["avg_coverage"]
        structured_stats_results["Coverage"]["std"][group] = stats["std_coverage"]
        structured_stats_results["Coverage"]["median"][group] = stats["median_coverage"]
        structured_stats_results["Coverage"]["mode"][group] = stats["mode_coverage"]
        structured_stats_results["Coverage"]["var"][group] = stats["var_coverage"]

        # Clarity
        structured_stats_results["Clarity"]["avg"][group] = stats["avg_clarity"]
        structured_stats_results["Clarity"]["std"][group] = stats["std_clarity"]
        structured_stats_results["Clarity"]["median"][group] = stats["median_clarity"]
        structured_stats_results["Clarity"]["mode"][group] = stats["mode_clarity"]
        structured_stats_results["Clarity"]["var"][group] = stats["var_clarity"]

        # Edge & Negative Cases Score
        structured_stats_results["Edge & Negative Cases Score"]["avg"][group] = stats[
            "avg_edge_and_negative_cases_score"
        ]
        structured_stats_results["Edge & Negative Cases Score"]["std"][group] = stats[
            "std_edge_and_negative_cases_score"
        ]
        structured_stats_results["Edge & Negative Cases Score"]["median"][group] = (
            stats["median_edge_and_negative_cases_score"]
        )
        structured_stats_results["Edge & Negative Cases Score"]["mode"][group] = stats[
            "mode_edge_and_negative_cases_score"
        ]
        structured_stats_results["Edge & Negative Cases Score"]["var"][group] = stats[
            "var_edge_and_negative_cases_score"
        ]

        # Non-Functional Coverage
        structured_stats_results["Non-Functional Coverage"]["avg"][group] = stats[
            "avg_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["std"][group] = stats[
            "std_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["median"][group] = stats[
            "median_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["mode"][group] = stats[
            "mode_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["var"][group] = stats[
            "var_non_functional_coverage"
        ]

        # Overall Quality Score
        structured_stats_results["Overall Quality Score"]["avg"][group] = stats[
            "avg_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["std"][group] = stats[
            "std_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["median"][group] = stats[
            "median_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["mode"][group] = stats[
            "mode_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["var"][group] = stats[
            "var_quality_score"
        ]

    return structured_stats_results


# Function to create and save visualizations from performance metrics data
def create_performance_charts(data, stats_test_data, output_dir="charts"):
    os.makedirs(output_dir, exist_ok=True)

    # #########################################################
    # Visualizing Descriptive Statistics from Test Results Data
    # #########################################################

    # Set style
    plt.style.use("default")

    # 1. Key Performance Metrics Bar Charts
    metrics = data["Key Performance Metrics"]
    for metric, values in metrics.items():
        plt.figure(figsize=(12, 6))
        bars = plt.bar(values.keys(), values.values())
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{metric} by Model")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric.lower().replace(" ", "_")}.png')
        plt.close()

    # 2. Coverage Statistics Comparison
    coverage_stats = data["Coverage"]
    metrics = ["avg", "std", "median", "mode", "var"]

    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = coverage_stats[metric]
        plt.bar(values.keys(), values.values())
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f"Coverage {metric.upper()}")
        plt.tight_layout()
    plt.savefig(f"{output_dir}/coverage_statistics.png")
    plt.close()

    # 3. Clarity Statistics Comparison
    clarity_stats = data["Clarity"]
    metrics = ["avg", "std", "median", "mode", "var"]

    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = clarity_stats[metric]
        plt.bar(values.keys(), values.values())
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f"Clarity {metric.upper()}")
        plt.tight_layout()
    plt.savefig(f"{output_dir}/clarity_statistics.png")
    plt.close()

    # 4. Edge & Negative Cases Statistics Comparison
    edge_cases_stats = data["Edge & Negative Cases Score"]
    metrics = ["avg", "std", "median", "mode", "var"]

    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = edge_cases_stats[metric]
        plt.bar(values.keys(), values.values())
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f"Edge & Negative Cases {metric.upper()}")
        plt.tight_layout()
    plt.savefig(f"{output_dir}/edge_cases_statistics.png")
    plt.close()

    # 5. Non-Functional Coverage Statistics Comparison
    nf_coverage = data["Non-Functional Coverage"]
    metrics = ["avg", "std", "median", "mode", "var"]

    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = nf_coverage[metric]
        plt.bar(values.keys(), values.values())
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f"Non-Functional Coverage {metric.upper()}")
        plt.tight_layout()
    plt.savefig(f"{output_dir}/nf_coverage_statistics.png")
    plt.close()

    # #########################################################
    # Visualizing Advance Statistics `ANOVA` & `t-Test` Results
    # #########################################################

    tests = stats_test_data.get("pairwise_t_tests", {})
    records = []
    models_set = set()
    for comp, result in tests.items():
        # comp looks like: "modelA_vs_modelB"
        if "_vs_" not in comp:
            continue
        modelA, modelB = comp.split("_vs_")
        models_set.update([modelA, modelB])

        records.append(
            {
                "Comparison": comp,
                "Model A": modelA,
                "Model B": modelB,
                "t_stat": result["t_stat"],
                "p_value": result["p_value"],
                "corrected_p_value": result["corrected_p_value"],
                "interpretation": result["interpretation"],
            }
        )

    df = pd.DataFrame(records)

    # ---------------------------------------------------------------------
    # 1) Create an Upper-Triangular Heatmap of Corrected p-values
    # ---------------------------------------------------------------------
    # Sort models so they appear in a consistent order
    models = sorted(models_set)

    # Create a matrix (DataFrame) for corrected p-values
    p_matrix = pd.DataFrame(
        np.ones((len(models), len(models))),  # Initialize with 1.0
        index=models,
        columns=models,
    )

    # Fill the matrix with the corrected p-values
    for _, row in df.iterrows():
        mA, mB = row["Model A"], row["Model B"]
        p_matrix.loc[mA, mB] = row["corrected_p_value"]
        p_matrix.loc[mB, mA] = row["corrected_p_value"]

    # We want an upper-triangular heatmap:
    # Mask the lower triangle so it is hidden
    mask = np.tril(np.ones_like(p_matrix, dtype=bool), k=-1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        p_matrix,
        mask=mask,
        cmap="RdYlBu_r",  # Red ~ low p, Blue ~ high p
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Corrected p-value"},
    )
    plt.title("Pairwise Corrected p-values Heatmap")
    plt.xlabel("Models")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pairwise_corrected_pvalues_heatmap.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # 2) Horizontal Bar Chart of t-statistics, colored by significance
    # ---------------------------------------------------------------------
    # Sort comparisons by t_stat so bars go from most negative to most positive
    df_sorted = df.sort_values("t_stat", ascending=True).reset_index(drop=True)

    # Define colors: red if "Significant", grey if "Not significant"
    bar_colors = [
        "red" if row["interpretation"].lower() == "significant" else "grey"
        for _, row in df_sorted.iterrows()
    ]

    plt.figure(figsize=(8, 16))
    plt.barh(df_sorted["Comparison"], df_sorted["t_stat"], color=bar_colors)
    plt.axvline(0, color="black", linewidth=0.8)  # Reference line at t=0

    # Create a legend for significance
    red_patch = mpatches.Patch(color="red", label="Significant")
    grey_patch = mpatches.Patch(color="grey", label="Not significant")
    plt.legend(handles=[red_patch, grey_patch], title="Significance")

    plt.xlabel("t-statistic")
    plt.ylabel("Comparison")
    plt.title("Pairwise t-test Results: t-statistics")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pairwise_tstats_bar_chart.png", dpi=300)
    plt.close()
