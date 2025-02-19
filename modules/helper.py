import json
import datetime
import time
import pandas as pd
from langchain.schema import AIMessage
import json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AIMessage):
            return {
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs,
                "type": "AIMessage",
            }
        return super().default(obj)


# Loads data from a JSON file
def load_data(json_file: str):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return True, data if isinstance(data, list) else []  # Ensure data is a list
    except (FileNotFoundError, json.JSONDecodeError):
        return False, []  # Return an empty list instead of None


# Chunk large data into smaller pieces
def chunk_data(data: list, chunk_size: int):
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


# In order to avoid losing status of successfully evaluated cases and
# unsuccessful cases due to any crash (i.e. code, server, internet, etc...)
# This function is to save each iteration into their respective JSON files
# based on successfully and unsuccessfully evaluated state
def save_data(data: list, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=CustomEncoder)


# Return a dictionary with formatted start, end times and duration
def format_time_info(start: datetime, end: datetime):
    return {
        "start_time": start.strftime("%d/%m/%Y %H:%M:%S"),
        "end_time": end.strftime("%d/%m/%Y %H:%M:%S"),
        "duration_in_sec": (end - start).total_seconds(),
    }


# Function to filter out already processed test cases
def filter_unprocessed_test_cases(all_test_cases: list, processed_test_cases: list):
    # Create a set of (test_case_id, group) tuples for quick lookup
    processed_set = {
        (case["test_case_id"], case["group"]) for case in processed_test_cases
    }

    # Include only those test cases whose (test_case_id, group) is not in the processed_set
    return [
        case
        for case in all_test_cases
        if (case["test_case_id"], case["group"]) not in processed_set
    ]


# Function to count no of token utilized
def calculate_tokens(test_cases: list):
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for case in test_cases:
        usage_metadata = case.get("usage_metadata", {})
        total_input_tokens += usage_metadata.get("input_tokens", 0)
        total_output_tokens += usage_metadata.get("output_tokens", 0)
        total_tokens += usage_metadata.get("total_tokens", 0)

    return total_input_tokens, total_output_tokens, total_tokens


# Function to print rate limit info
def rate_limit_logger(
    condition: str, active_api_key: int, request_made: int, token_made: int
):
    line = "-" * 30
    print(line)
    print(f"Under {condition} Condition Block:")
    print(line)
    print(f"\nPer {condition} Limit Exceeded:")
    print("-" * 28)
    print("Active API Key:", active_api_key)
    print(f"RP{condition[0]}:", request_made)  # RPD or RPM
    print(f"TP{condition[0]}:", token_made)  # TPD or TPM
    print(f"Switching API Key - Resetting {condition} Counters")
    print("-" * 28, "\n")


# Function to extract data from specified sheet in the input file and save it to output file.
def extract_sheet_data(input_filename: str, sheet_name: str):
    try:
        data = pd.read_excel(input_filename, sheet_name=sheet_name).to_dict(
            orient="records"
        )
        return data

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Function to clean raw data and convert into a processable format
def clean_raw_dataset(data_set, group_name):
    return [
        {
            # Extracting required fields, defaulting to 'NA' if missing
            **{
                key: tc_item.get(key, "NA")
                for key in [
                    "test_case_id",
                    "software_name",
                    "software_desc",
                    "test_module",
                    "test_feature",
                    "test_case_title",
                    "test_case_description",
                    "pre_conditions",
                    "test_steps",
                    "test_data",
                    "expected_outcome",
                    "severity_status",
                ]
            },
            # Assigning group names based on the 'prompt_type' field
            "group": (
                tc_item["prompt_type"].lower()
                if tc_item["prompt_type"] == "Human-Engineers"
                else f"{group_name}-{tc_item['prompt_type'].lower()}"
            ),
        }
        for tc_item in data_set  # Iterating over all test cases
    ]


# Function to rotate API key
def rotate_api_key(active_key, total_keys):
    if (active_key + 1) > total_keys:
        return 0
    else:
        return active_key + 1


# Function to handle rate limit and API key rotation
def handle_rate_limit(limit_type, active_key, requests, tokens, api_keys):
    rate_limit_logger(limit_type, active_key, requests, tokens)
    new_key = rotate_api_key(active_key, len(api_keys))
    return new_key, 0, 0  # Returns new key and reset counters


# Function to save progress checkpoint every N iterations (e.g., 100)
def save_checkpoint(i, save_after_n_iteration, success_jobs, failed_jobs, base_path):
    if i % save_after_n_iteration == 0:
        save_data(success_jobs, f"{base_path}success.json")
        save_data(failed_jobs, f"{base_path}failed.json")
