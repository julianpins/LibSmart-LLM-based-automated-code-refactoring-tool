# LibSmart: AI-Powered Numpy Code Modernization

**LibSmart** is a developer tool that automatically refactors deprecated NumPy code using fine-tuned Large Language Models (LLMs). It combines a powerful model backend with an intuitive VS Code extension and integrated Retrieval Augmented generation (RAG) for high accuracy suggestions.

---

## Repository Structure
```
.
├── data/                  Training and evaluation data, ChromaDB setup and scraping scripts
├── evaluation/            Evaluation scripts, detailed results and summary
├── extension/             VS Code extension source code (frontend)
├── fine-tuning/           Fine-tuned models and training scripts
├── presentation_report/   Final project report and presentation
├── server/                Backend server (FastAPI, RAG pipeline)
├── demo.py                Standalone demo script for testing
└── requirements.txt       Python dependencies
```

---

## Getting Started
Follow these steps to set up and run the project locally.
### Prerequisites
- Python 3.9+
- [Visual Studio Code](https://code.visualstudio.com/)

### Installation
1. **Clone the repository:**
    ```bash
        git clone [https://gitlab.lrz.de/bpc-ss-25/mcmurdo.git]
        cd mcmurdo
    ```

2. **Install dependencies**
    ```bash
        pip install -r requirements.txt
    ```
    >**Troubleshooting Note:** On windows you might encouter a 'llama.dll' error. If so, run the following command to reinstall the necessary package:
    >```bash
    >pip install llama-cpp-python --force-reinstall --no-cache-dir
    >```
---

## Running the Application
1. **Start the Backend Server:**
    From the root directory, run the server. This launches the model API and the RAG pipeline.
    ```bash
    python server/main.py
    ```
    You will be prompted to select a model.
    ```
    Available models:
    1. gemma-2-2b-it-numpy-refactor-v1_q4_k_m
    2. tinyllama-1.1b-chat-numpy-refactor-v2_q4_k_m
    Select model (number):
    ```
    > **Model Performance**: The fine-tuned **Gemma 2B** is quite good at avoiding unnecessary changes and provides more reliable context. **TinyLlama 1.1B** is faster but may occasionally hallucinate deprecations.

2. **Launch the VS Code extension**
    - Open the project folder in VS Code and open extension/extension.js
    - Press Fn + F5 or navigate to the "Run and Debug" panel (Ctrl+Shift+D) and select "Extension Development Host"

    This will open a new "Extension development Host" window where the LibSmart extension is active.

    > **Developer Tip**: To test the extension's UI without running the backend, set 'useMock = true;' in 'extension/webview/main.js'.
---

## How to Use
Once the backend is running and the Extension Development Host is open:
1. Open or create a Python script containing NumPy code.
2. Select a code snippet you wish to modernize.
3.  Right-click and choose **"Modernize NumPy code"** from the context menu, or use the hotkey `Ctrl+M` (`Cmd+M` on macOS).
4. The LibSmart side panel will open with a suggested change and an explanation. If no deprecated usage is found, the message "No deprecated functionality found" will appear.
5. Click **"Accept Change"** to replace the original code in the editor.

---

## Evaluation
To reproduce our evaluation results:
1. Ensure the backend server is running with your chosen model.
2. Run the evaluation script:
```bash
    python evaluation/scripts/rag_evaluation.py
```
The script will process the validation dataset and print model outputs. Aggregated results will be saved to the `/evaluation` directory as `.csv` files.

---

## Roadmap
- Applying Bayesian optimization to streamline the fine-tuning process
- Adding support for other libraries like Pandas and Polars.
- Expanding the ChromaDB for more comprehensive context retrieval

---

## Authors
- Priyam Turakhia
- Julian Pins
- Frytz Martinek
- Alena Bobovich