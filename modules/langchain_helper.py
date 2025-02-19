from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import Optional

# ----------------------------
# 1. PYDANTIC MODEL DEFINITION
# ----------------------------
# Define the Pydantic models to match the desired JSON structure
class ScoreComponent(BaseModel):
    score: int = Field(description="Numerical score between 1 and 5")
    reason: Optional[str] = Field(description="Detailed reasoning explaining the assigned score")

    @field_validator('score')
    def validate_score_range(cls, value):
        if not 1 <= value <= 5:
            raise ValueError("Score must be between 1 and 5")
        return value

class Evaluation(BaseModel):
    coverage: ScoreComponent = Field(description="Evaluation of test coverage")
    clarity: ScoreComponent = Field(description="Evaluation of test clarity")
    edge_and_negative_cases_score: ScoreComponent = Field(
        description="Evaluation of edge and negative test cases"
    )
    non_functional_coverage: ScoreComponent = Field(
        description="Evaluation of non-functional requirement coverage"
    )
    justification: Optional[str] = Field(description="Overall evaluation justification")

class TestCaseEvaluation(BaseModel):
    test_case_id: str = Field(description="Unique identifier for the test case")
    evaluation: Evaluation = Field(description="Complete evaluation details")

# Initialize the PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=TestCaseEvaluation)

# Dictionary to make the prompt template and input variables collapsible for better readability
PROMPT = {
    "INPUT_VARIABLES": [
        "software_name", "software_desc", "test_case_id", "test_module", "test_feature", 
        "test_case_title", "test_case_description", "pre_conditions", "test_steps", 
        "test_data", "expected_outcome", "severity_status"
    ],

    "TEMPLATE": """
    You are a Senior QA Analyst with 20+ years of experience in rigorous
    test case evaluation. Use the following structured approach and instructions
    carefully to ensure a detailed and accurate evaluation.

    **Chain-of-Thought Analysis**
    Follow these steps sequentially:
    1. **Understand Context:** Deeply analyze the test case and software context.
    2. **Break Down Reasoning:** For each criterion, follow this format:
    - Analyze the test case details.
    - Identify strengths and weaknesses.
    - Assign a score based on the scoring scale.
    3. **Apply Industry Standards:** Compare against ISTQB, IEEE 291, and OWASP Top 10 standards.
    4. **Resolve Conflicts:** Use severity analysis to resolve contradictions (e.g., prioritize critical issues over minor ones).
    5. **Self-Consistency Check:** Validate scores against similar test cases and ensure reasoning aligns with industry best practices.

    **Evaluation Criteria:**
    1. **Coverage (30% Weight):**
    - Sub-criteria: Module coverage, feature coverage, integration points.
    - Ensure a sufficient portion of the system is exercised.
    - Why 30%? Coverage directly impacts the reliability of testing.

    2. **Clarity (20% Weight):**
    - Sub-criteria: Readability, maintainability, logical flow.
    - Ensure the test case is easy to understand and maintain.
    - Why 20%? Clear test cases reduce maintenance effort and improve collaboration.

    3. **Edge & Negative Cases (25% Weight):**
    - Sub-criteria: Uncommon scenarios, error-prone conditions, failure modes.
    - Check for uncommon or error-prone conditions.
    - Why 25%? Edge cases often reveal critical bugs.

    4. **Non-Functional Coverage (25% Weight):**
    - Sub-criteria: Performance, usability, security, scalability.
    - Evaluate non-functional aspects of the system.
    - Why 25%? Non-functional requirements are crucial for user satisfaction and system robustness.

    After evaluating test case thoroughly and meticulously, assign scores to each
    criteria based on the evaluation. Using the following scoring scale:

    **Scoring Scale (1 to 5):**
    5 = Exceeds standards, benchmark example
    4 = Meets standards with minor improvements
    3 = Requires moderate rework
    2 = Needs major overhaul
    1 = Complete redesign needed

    Now, evaluate the following test case based on the instructions, criteria, and
    scoring scale given above and provide detailed justification for your evaluation:

    **Test Case Details:**
    - Test Case ID: {test_case_id}
    - Software Name: {software_name}
    - Software Description: {software_desc}
    - Test Module: {test_module}
    - Test Feature: {test_feature}
    - Test Case Title: {test_case_title}
    - Test Case Scenario/Description: {test_case_description}
    - Pre-conditions: {pre_conditions}
    - Test Steps: {test_steps}
    - Test Data: {test_data}
    - Expected Outcome: {expected_outcome}
    - Severity: {severity_status}

    **JSON format Output:**
    After evaluating the test case thoroughly, provide **ONLY** your evaluation as a JSON object
    enclosed in triple backticks (`json ... `). **DO NOT INCLUDE ANY ADDITIONAL TEXT**
    outside the JSON block. Your response **MUST ONLY CONTAIN** the JSON, and nothing else.

    Following JSON Schema represents the format I want you to follow to generate your answer.
    Now, generate a JSON object that will contain the following information:

    ```json
    {{
        "test_case_id": "{test_case_id}",
        "evaluation": {{
            "coverage": {{
            "score": 5,
            "reason": "The test case covers all critical modules and features, including edge cases."
            }},
            "clarity": {{
            "score": 4,
            "reason": "The steps are clear, but the expected outcome could be more detailed."
            }},
            "edge_and_negative_cases_score": {{
            "score": 3,
            "reason": "Some edge cases are missing, such as handling null inputs."
            }},
            "non_functional_coverage": {{
            "score": 2,
            "reason": "No mention of performance or security testing."
            }},
            "justification": "Overall, the test case is strong in functional coverage but lacks depth in non-functional aspects."
        }}
    }}
    ```
    """,
}

# Initializing the PromptTemplate with input variables and partial variables
prompt = PromptTemplate(
    input_variables=PROMPT["INPUT_VARIABLES"],
    template=PROMPT["TEMPLATE"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ---------------------
# 2. GROQ CHAIN MAKER
# -------------------
# Function to return the chain after chaining of prompt and model
def get_groq_chain(model_name, api_key):
    # Loading model
    model = ChatGroq(api_key=api_key, 
                     model=model_name, 
                     max_tokens=500, 
                     max_retries=2)

    # Chaining the prompt and model
    chain = prompt | model

    return chain

