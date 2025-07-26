# LibSmart – NumPy Code Modernization Assistant

This project fine-tunes open-source LLMs to automatically refactor deprecated NumPy code. It combines a model backend with a VS Code extension frontend and integrates retrieval-augmented generation (RAG) for improved context understanding.

---

## Project Structure
- data/: Training & evaluation data, ChromaDB setup, scraping scripts
- evaluation/: Model evaluation scripts and result visualizations
- extension/: VS Code extension (frontend UI and interaction logic)
- fine-tuning/: Fine-tuned models and training scripts
- presentation_report/: Final report and presentation slides
- server/: Backend server (model API + RAG pipeline)
- requirements.txt: Python dependencies for this project
- demo.py: Example code snippet for extension testing


---

## Installation

Install all necessary Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the project
1. From the root directory, run the backend service:

```bash
python server/main.py´
```

This will launch the model server and RAG pipeline.

2. Launch the VS Code extension
Open the extension/ folder in Visual Studio Code.

Press Ctrl + F5 or go to Run and Debug → "Extension Development Host".

This will open a new VS Code window with the extension running.

---

## How to Use
Once the Extension Development Host is open:

- Open any Python file with NumPy code.

- Select a code snippet you'd like to modernize.

- Right-click and select: Modernize NumPy code

A side panel will open:

- If deprecated code is detected, a modernized version with an explanation will be shown.

- If no deprecated usage is found, the message "No deprecated functionality found" will appear.

- Click Accept Change to replace the original code in the editor.