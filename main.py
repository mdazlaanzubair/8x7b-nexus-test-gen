import time
import sys
import datetime
import json
from tabulate import tabulate
from modules.langchain_helper import get_groq_chain, parser

# Custom imports
from modules.helper import load_data, save_data, filter_unprocessed_test_cases
from modules.helper import format_time_info, rate_limit_logger, clean_raw_dataset
from modules.helper import calculate_tokens, extract_sheet_data, save_checkpoint
from modules.helper import rotate_api_key, handle_rate_limit
from modules.api_keys import api_keys

# 1. Checking API keys import in order to proceed further
# ################################################
if len(api_keys) <= 0:
    print("No API Key Found. Load API Keys list")
    exit()


# 2. Loading test cases data
# ################################################
is_exist, test_cases = load_data("dataset/processed_data/test_cases.json")
if not is_exist:
    # Extracting and cleaning data from excel sheets
    input_filename = "dataset/raw_data/raw_data.xlsx"

    test_cases = []
    sheet_mapping = {
        "human_test_cases": "human-engineers",
        "gpt4_test_cases": "gpt4-model",
        "o1_test_cases": "o1-model",
    }

    for sheet, group in sheet_mapping.items():
        raw_data = extract_sheet_data(input_filename, sheet)
        if raw_data:
            test_cases.extend(clean_raw_dataset(raw_data, group))

    print("Data Extracted & Cleaned Successfully")

    # Save cleaned test cases to JSON
    save_data(test_cases, "dataset/processed_data/test_cases.json")
    print("Data Saved Successfully")


# 3. Filtering out test cases that are already successfully processed
# ################################################
is_exist, success_test_cases = load_data("dataset/evaluation_data/success.json")
remaining_test_cases = test_cases
if is_exist:
    remaining_test_cases = filter_unprocessed_test_cases(test_cases, success_test_cases)


# 4. Logging insights before starting the evaluation
# ################################################
status_data = [
    ["Total Cases", len(test_cases)],
    ["Processed Cases", len(success_test_cases)],
    ["Remaining Cases", len(remaining_test_cases)],
]

print("\n\nINITIAL STATUS\n" + "-" * 30)

# Display Initial Status
print(tabulate(status_data, tablefmt="fancy_grid", colalign=("left", "right")))

# Calculate token utilization
total_input_tokens, total_output_tokens, total_tokens = calculate_tokens(
    success_test_cases
)
token_data = [
    ["Input Tokens", total_input_tokens],
    ["Output Tokens", total_output_tokens],
    ["Total Tokens", total_tokens],
]

# Display Token Utilization
print(tabulate(token_data, tablefmt="fancy_grid", colalign=("left", "right")))


# 5. Exit if there is not test case to process
# ################################################
if len(remaining_test_cases) <= 0:
    print("\nNo test case to process. Exiting...")
    exit()


# 6. Proceed with the evaluation
# ################################################

# Initializing global variables to be used in the script

# Rate limit constants
REQUEST_PER_MINUTE = 30
REQUEST_PER_DAY = 14400
TOKENS_PER_MINUTE = 5000
TOKENS_PER_DAY = 500000
DELAY = 500000

# Active model and API key
models_list = ["llama3-70b-8192", "mixtral-8x7b-32768", "qwen-2.5-32b"]
active_model = models_list[1]
active_api_key = 0

# Counters to keep track of requests and tokens
request_made_per_minute = 0
request_made_per_day = 0
token_made_per_minute = 0
token_made_per_day = 0

# Lists to store success and failed jobs
# (job mean test cases evaluation task)
success_jobs = []
failed_jobs = []


# Function to evaluate a test case one-by-one
# using the specified model and API key
def evaluate_test_case(test_case, model_name, api_key):

    # Using global variables
    global request_made_per_minute, request_made_per_day
    global token_made_per_minute, token_made_per_day
    global success_jobs, failed_jobs

    # Get the GROQ chain for the model
    chain = get_groq_chain(model_name, api_key)
    start_time = datetime.datetime.now()
    llm_raw_output = ""

    # Prepare input without modifying original test_case
    input_variables = {k: v for k, v in test_case.items() if k != "group"}

    # Evaluate the test case
    try:
        # Invoke the chain
        llm_raw_output = chain.invoke(input_variables)
        end_time = datetime.datetime.now()

        # Extract content
        content = llm_raw_output.content

        # Extract metadata
        response_metadata = llm_raw_output.response_metadata
        usage_metadata = llm_raw_output.usage_metadata

        # Parse the response
        parsed_response = parser.parse(content)

        # Dumping parsed response to a dictionary
        success_case = parsed_response.model_dump()

        # Update success case with additional metadata
        success_case.update(
            {
                "evaluated_by": model_name,
                "time_taken": format_time_info(start_time, end_time),
                "group": test_case["group"],
                "test_case_id": test_case["test_case_id"],
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
            }
        )

        # Update global counters
        tokens_used = usage_metadata["total_tokens"]
        request_made_per_minute += 1
        request_made_per_day += 1
        token_made_per_minute += tokens_used
        token_made_per_day += tokens_used

        # Append to success jobs
        success_jobs.append(success_case)

    # Handle exceptions
    except Exception as e:
        # Check for invalid API key
        if "invalid_api_key" in str(e):
            print("Invalid API Key. Please check your API key configuration.")
            print("Test Case ID:", test_case["test_case_id"])
            print("API Key:", api_key)
            sys.exit(1)  # Use sys.exit() with an exit code

        # Prepare failed case
        failed_case = {
            "evaluated_by": model_name,
            "test_case": test_case,
            "llm_raw_output": llm_raw_output,
            "error_exception_details": str(e),
            "time_taken": format_time_info(start_time, datetime.datetime.now()),
        }

        # Append to failed jobs
        failed_jobs.append(failed_case)


# Function to handle rate limit and API key rotation
def handle_rate_limit(
    active_api_key, max_req_limit, max_token_limit, req_count, token_count
):
    # Logging rate limit info
    rate_limit_logger("Minute", active_api_key, req_count, token_count)


# ################################################
# MAIN EXECUTION
# ################################################

# Looping through remaining test cases
for i, test_case in enumerate(remaining_test_cases):
    print(f"\n{i} - Evaluation Started - Test Case ID: {test_case['test_case_id']}")

    # Check minute-based rate limits
    if (
        request_made_per_minute >= REQUEST_PER_MINUTE
        or token_made_per_minute >= TOKENS_PER_MINUTE
    ):
        active_key, req_per_min, token_per_min = handle_rate_limit(
            "Minute",
            active_key,
            request_made_per_minute,
            token_made_per_minute,
            api_keys,
        )

    # Check day-based rate limits
    elif (
        request_made_per_day >= REQUEST_PER_DAY or token_made_per_day >= TOKENS_PER_DAY
    ) and active_key < len(api_keys) - 1:
        active_key, req_per_day, token_per_day = handle_rate_limit(
            "Day", active_key, request_made_per_day, token_made_per_day, api_keys
        )

    # Process the current test case with the active API key
    current_api_key = api_keys[active_api_key]
    evaluate_test_case(test_case, active_model, current_api_key)
    print(f"\n{i} - Evaluation Completed - Test Case ID: {test_case['test_case_id']}")

    time.sleep(1)  # Anti-spam delay

    # Save checkpoints after every N iterations (e.g., 30)
    checkpoint_path = "dataset/evaluation_data/"
    save_checkpoint(i, 30, success_jobs, failed_jobs, checkpoint_path)


# Final Report
total_cases = len(test_cases)
success_cases = len(success_jobs)
failed_cases = len(failed_jobs)
remaining_cases = total_cases - success_cases  # considering failed jobs as remaining

report_data = [
    ["Remaining", remaining_cases, f"{(remaining_cases / total_cases) * 100:.2f}%"],
    ["Success", success_cases, f"{(success_cases / total_cases) * 100:.2f}%"],
    ["Failed", failed_cases, f"{(failed_cases / total_cases) * 100:.2f}%"],
    ["Total", total_cases, "100.00%"],
]

print("\n" + "=" * 35)
print("           FINAL REPORT           ")
print("=" * 35)
print(tabulate(report_data, headers=["Status", "Count", "Percentage"], tablefmt="grid"))
