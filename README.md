# 8x7B Nexus: AI, Prompts, & Human QA for TestGen

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides the codebase, datasets, and resources used to conduct this comparative evaluation and replicate the findings in research paper titled: **8x7B Nexus: Converging AI Reasoning, Prompt Engineering, and Human QA in Test Generation.**

> This research investigates the reasoning capabilities of large language models (LLMs) in the context of software test case generation.  We specifically evaluate the generated test cases by all 12 experimental groups in this research using  [**Mixtral-8x7b**](https://mistral.ai/en/news/mixtral-of-experts), a Sparse Mixture of Experts model. 

> [**Mixtral 8x7B’s architecture**](https://datasciencedojo.com/blog/mixtral-of-experts-by-mistral-ai/#:~:text=Distinctive%20for%20its%20use%20of,of%20language%20with%20notable%20precision.)—with its selective expert routing, high temporal locality, computational efficiency, and robust multilingual performance—makes it a strong candidate for generating and evaluating comprehensive test cases.

The research addresses the following key questions:

1.  **RQ1: Reasoning Differentiation:** Do the reasoning capabilities of an AI Model truly differentiate it from other AI models lacking such capabilities (like GPT-4o) in test case generation?

2.  **RQ2: Reasoning for Test Case Challenges:** Can o1 model's reasoning abilities effectively address the inherent challenges in generating comprehensive and high-quality test cases?

3.  **RQ3: Prompt Engineering for Performance Gap:** Can prompt engineering techniques help narrow the performance gap in test case generation between models with built-in chain-of-thought reasoning (like Open AI's o1) and those without (like GPT-4o)?

4.  **RQ4: AI vs. Human QA Productivity and Quality:**  How does the productivity and quality of AI-generated test cases (from o1 and GPT-4o) compare to those produced by experienced Human QA Engineers?

## Folder Structure

```folder_structure
8x7b-nexus-test-gen/
├── dataset/
│   ├── raw_data/           # Excel sheet contain test cases, software and prompts
│   ├── processed_data/     # Cleaned copy of test cases in a processable format
│   ├── evaluation_data/    # Test cases data after getting scored and evaluation
│   └── results_data/       # Holds result of test cases score given by AI model
├── modules/                # Helper function for Python scripts
├── main.py                 # Main script to start evaluation using AI
├── stats_calc.py           # Script to calculate results on test cases score
├── requirements.txt        # List of required Python packages
├── LICENSE                 # Licensing information (MIT License)
└── README.md               # This documentation file
```

## Getting Started

To run this script, you'll need to set up your environment with the necessary dependencies and models. Following are the libraries that this Python project require to evaluate the quality of test cases:

- [**LangChain**](https://github.com/hwchase17/langchain)

- [**langchain-groq**](https://python.langchain.com/v0.1/docs/integrations/chat/groq/)

- [**pandas**](https://pandas.pydata.org/)

- [**numpy**](https://numpy.org/)

- [**matplotlib**](https://matplotlib.org/)

- [**seaborn**](https://seaborn.pydata.org/)

### Prerequisites

- **Python 3.x**: [**Download Python**](https://www.python.org/downloads/)

- **Git**: [**Download Git**](https://git-scm.com/downloads)

- **VS Code** (Optional but recommended): [**Download VS Code**](https://code.visualstudio.com/)

- **langchain-groq**: In case you wanted to use **`Groq API`**, refer this [**official documentation**](https://console.groq.com/docs/quickstart) from [**Groq Cloud**](https://groq.com/).

## Installation and Setup

To replicate the experiments and analysis from the research paper, please follow these steps:

### 1. Clone the Repository

#### Open your terminal and run:

```bash
git clone https://github.com/mdazlaanzubair/8x7b-nexus-test-gen.git
```

```bash
cd 8x7b-nexus-test-gen
```

### 2. Create a Virtual Environment

#### It is recommended to use a virtual environment to manage dependencies.

- **On Windows:**

    **Create/Activate the virtual environment:**

    ```bash
    python -m venv env
    ```

    ```bash
    .\env\Scripts\activate
    ```

- **On macOS/Linux:**

    **Create/Activate the virtual environment:**

    ```bash
    python3 -m venv env
    ```

    ```bash
    source env/bin/activate 
    ```

### 3. Install Dependencies

#### If a `requirements.txt` file is available, install the dependencies by running:

- **On Windows:**

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can install the necessary packages manually:

    ```bash
    pip install langchain langchain-groq pandas numpy matplotlib seaborn scipy openpyxl tabulate
    ```

- **On macOS/Linux:**

    ```bash
    pip3 install -r requirements.txt
    ```

    Alternatively, you can install the necessary packages manually:

    ```bash
    pip3 install langchain langchain-groq pandas numpy matplotlib seaborn scipy openpyxl tabulate
    ```

### 4. Run the Project

Ensure your **`8x7b-nexus-test-gen/dataset/raw_data/`** has the test case data excel sheet i.e. **`test_cases.xlsx`**, and the following command:

> NOTE: Kindly ensure to import your API Keys when using Groq Cloud

* **Steps**:

    ```python 
    # 1_ Rename the `example_api_key.pys` file to `api_keys.py` in the `modules/` folder
    # 2_ Paste all your API Keys into it
    ```

* **Run** the **`main`** script with:

    **On Windows:**

    ```bash
    python main.py
    ```

    **On macOS/Linux:**
        
    ```bash
    python3 main.py
    ```

* **Run** the **`stats_calc.py`** script with:

    **On Windows:**

    ```bash
    python stats_calc.py
    ```

    **On macOS/Linux:**
        
    ```bash
    python3 stats_calc.py
    ```    

## Data Description

### The `dataset/` directory contains:

* **`raw_data/`**

    - **`raw_data.xlsx`**: Contains data regarding `software` and `prompts` used while generating `test cases` generated by **Human QA Engineers**, **GPT 4o**, and **o1** AI models.

    - **`open_ai_chat.zip`**: Contains chats data from which the models are used to generate test case of each software.

* **`processed_data/`**

    - **`test_cases.json`**: Contains cleaned copy of test cases extracted from raw data excel sheet in a processable format *(removed unnecessary fields)*.

* **`evaluation_data/`**

    - **`failed.json`**: Contains test cases that are failed during evaluation due to any exception with the error message.

    - **`success.json`**: Contains test cases with their respective scores give by AI model during evaluation in each segment *(i.e. Coverage, Clarity, Edge & Negative Tests, and Non-functional Coverage)*

* **`results_data/`**

    - **`charts/`**: Contains visuals of calculated stats in the form of charts.

    - **`score_result.json`**: Contains discrete statistics results that was performed to evaluate the performance of each group who wrote test cases.

    - **`stats_tests.json`**: Contains statistical analysis of quality of each group using **`ANOVA`** and **`T-Test`**.

    - **`data.zip`**: Contains data, results and chart that processed or produced while we were conducting this research.

## Troubleshooting

- **Virtual Environment:**
Ensure that your virtual environment is activated. You should see (venv) in your terminal prompt.

- **Dependency Installation:**
Verify that you have a stable internet connection and the necessary permissions to install packages.

- **LangChain / Langchain-Groq Setup:**
Make sure that you've obtained your [**API Keys**](https://console.groq.com/keys) from [**Groq Cloud**](https://groq.com/). Refer to the [**langchain-ollama documentation**](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.ollama.Ollama.html "langchain-ollama documentation") for additional guidance.

In case you wanted to use **`Groq API`**, refer this [**official documentation**](https://console.groq.com/docs/quickstart) from [**Groq Cloud**](https://groq.com/).

## Citation

#### If you utilize this repository or its contents in your research, please cite our work as follows:

```
[Citation details will be added after publication of the paper]
```

## License

This project is licensed under the **MIT License**. Please refer to the [**LICENSE**](https://opensource.org/license/MIT) for details.

## Contact
For any questions, suggestions, or issues, please open an issue in this repository or contact mdazlaan1996@gmail.com.