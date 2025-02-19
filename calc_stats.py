import json

# Custom modules
from modules.stats_helper import get_descriptive_stats, structuring_stats_in_metrics
from modules.stats_helper import create_performance_charts, perform_statistical_tests
from modules.stats_helper import format_data_to_evaluations
from modules.helper import load_data, save_data


BASE_REPOSITORY = "dataset/results_data/"

# Load data from processed_results.json
file_path = "dataset/evaluation_data/success.json"
is_exist, test_cases_scores = load_data(file_path)
if not is_exist or len(test_cases_scores) == 0:
    print("❌ No data found.")
    print("Please run the evaluation script first i.e. `main.py`")
    exit()


# Converting scores to evaluations and calculating
# weighted quality score of each test case
evaluations = format_data_to_evaluations(test_cases_scores)


if __name__ == "__main__":
    print("\n\nDescriptive Statistics:")
    print("-----------------------")

    results_descriptive = get_descriptive_stats(evaluations)
    print("  ✅ Calculated")

    structured_results_descriptive = structuring_stats_in_metrics(results_descriptive)
    print("  ✅ Structured")

    results_file_path = BASE_REPOSITORY + "score_result.json"
    save_data(structured_results_descriptive, results_file_path)
    print("  ✅ Saved\n")

    print("Test Statistics:")
    print("----------------")

    statistical_tests_results = perform_statistical_tests(evaluations, "quality_score")
    print("  ✅ Calculated")

    # Load the JSON data from the 'dataset' string
    structured_statistical_tests_results = json.loads(statistical_tests_results)
    print("  ✅ Converted to JSON")

    stats_tests_file_path = BASE_REPOSITORY + "stats_tests.json"
    save_data(structured_statistical_tests_results, stats_tests_file_path)
    print("  ✅ Saved\n")

    print("Creating Stats Visuals:")
    print("-----------------------")
    charts_directory = BASE_REPOSITORY + "charts"
    create_performance_charts(
        structured_results_descriptive,
        structured_statistical_tests_results,
        charts_directory,
    )
    print("  ✅ Charts saved\n")
