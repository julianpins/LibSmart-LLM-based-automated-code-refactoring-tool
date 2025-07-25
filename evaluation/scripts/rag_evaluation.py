import json
import os
import warnings
import textwrap
import ast
import re
import pandas as pd
import numpy as np
from numpy import ma
import pickle
import numpy_financial as npf
import requests
from pathlib import Path

VALIDATION_DATA_PATH = str(Path(__file__).parent.parent.parent / "data" / "datasets" / "validation_data.json")
RESULTS_PATH = str(Path(__file__).parent.parent)
DETAILED_RESULTS_CSV = RESULTS_PATH + "/evaluation_detailed.csv"
SUMMARY_METRICS_CSV = RESULTS_PATH + "/evaluation_summary.csv"

EVAL_GLOBALS = {
    'np': np,
    'ma': ma,
    'os': os,
    'ast': ast,
    'pickle': pickle,
    'npf': npf,
}

def clean_for_ast(code):
    return "\n".join(
        line.rstrip() for line in code.splitlines()
        if line.rstrip() and not line.strip().startswith("#")
    )

def compare_outputs(actual, expected):
    if isinstance(actual, np.ma.MaskedArray) and actual.ndim == 0: actual = actual.item() if not actual.mask else np.ma.masked
    if isinstance(expected, np.ma.MaskedArray) and expected.ndim == 0: expected = expected.item() if not expected.mask else np.ma.masked
    if actual is np.ma.masked and expected is np.ma.masked: return True
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual): return False
        return all(compare_outputs(a, e) for a, e in zip(actual, expected))
    if isinstance(actual, str) and isinstance(expected, str): return actual.replace(" ", "").replace("\n", "") == expected.replace(" ", "").replace("\n", "")
    is_actual_arraylike = isinstance(actual, (np.ndarray, list, tuple))
    is_expected_arraylike = isinstance(expected, (np.ndarray, list, tuple))
    if is_actual_arraylike and is_expected_arraylike:
        try:
            actual_arr, expected_arr = np.asarray(actual), np.asarray(expected)
            if any(np.issubdtype(arr.dtype, np.number) for arr in [actual_arr, expected_arr]): return np.allclose(actual_arr, expected_arr, equal_nan=True)
            return np.array_equal(actual_arr, expected_arr)
        except (ValueError, TypeError): return False
    if is_actual_arraylike != is_expected_arraylike: return False
    return actual == expected

def check_compiles(f, code):
    execution_scope = {}
    try:
        exec(code, EVAL_GLOBALS, execution_scope)
        compiled_function = execution_scope.get(f)
        if not callable(compiled_function):
            raise NameError(f"Function '{f}' was not defined correctly.")
        return True, "OK", compiled_function
    except Exception as e:
        return False, f"SyntaxError: {e}", None

def check_indentation(f, code):
    scope = {}
    try:
        exec(code, EVAL_GLOBALS, scope)
        scope.get(f)
        return True, "OK"
    except Exception as e:
        return False , f"IndentationError: {e}"

def check_no_deprecations(f, input):
    if not callable(f): return False, "Function not callable"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if isinstance(input, tuple): f(*input)
            else: f(input)
        except AttributeError as e:
            if "module 'numpy' has no attribute" in str(e):
                return False, "Output contained deprecated features"
        except Exception as e:
            return False, f"Error during execution, most likely due to deprecation: {e}"

        for item in w:
            if issubclass(item.category, DeprecationWarning):
                return False, "Output contained deprecated features"
    return True, "OK"

def check_functionality(fun, test_cases: list):
    for j, case in enumerate(test_cases):
        try:
            input = eval(case['input'], EVAL_GLOBALS)
            expected_output = eval(case['expected_output'], EVAL_GLOBALS)
            actual_output = fun(*input) if isinstance(input, tuple) else fun(input)
            if not compare_outputs(actual_output, expected_output):
                return False, f"Failed: [Expected: {repr(expected_output)}, Got: {repr(actual_output)}]"
        except Exception as e:
            return False, f"Error during execution of Test Case {j+1}: {e}"
    return True, "All Test Cases passed"

def parse_model_output2(raw_output):
    output = raw_output.split("<start_of_turn>model")[-1]
    code_match = re.search(r"```python\n(.*?)\n```", output, re.DOTALL)
    code = code_match.group(1) if code_match else ""
    if not code:
        return output
    return code

def call_rag_api(code, version):
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"code": code, "numpy_version": version},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

#evaluation
def main():
    print("Starting RAG evaluation script")
    
    # Check if API is available
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=60)
        if health_response.status_code != 200:
            print("RAG API is not available! Please start the server first.")
            return
        health_data = health_response.json()
        if not health_data.get("chroma_connected") or not health_data.get("model_available"):
            print("RAG service is not properly initialized!")
            return
        print("RAG API is ready")
    except Exception as e:
        print(f"Failed to connect to RAG API: {e}")
        return

    print(f"Loading validation data from: {VALIDATION_DATA_PATH}")

    if not os.path.exists(VALIDATION_DATA_PATH):
        print(f"Validation file not found! Please ensure '{VALIDATION_DATA_PATH}' exists.")
        return
    with open(VALIDATION_DATA_PATH, 'r') as f:
        validation_data = json.load(f)

    results_list = []
    total_samples = len(validation_data)

    for i, sample in enumerate(validation_data):
        print(f"\nProcessing sample {i+1}/{total_samples}:")
        try:            
            api_result = call_rag_api(sample['input'], sample['version'])            
            if api_result.get('error'):
                print(f"RAG API failed: {api_result['error']}")
                results_list.append({
                    'sample_index': i, 
                    'compiles': 'Fail: RAG API error',
                    'correct_indentation': 'Fail: Skipped',
                    'no_deprecations': 'Fail: Skipped', 
                    'correct_functionality': 'Fail: Skipped',
                    'functions_retrieved': 0,
                    'functions_with_context': 0
                })
                continue
                
            modernized_code = api_result.get('modernized_code', '')
            retrieved_context = api_result.get('retrieved_context', {})
            
        except Exception as e:
            print(f"RAG API error: {e}")
            results_list.append({
                'sample_index': i,
                'compiles': f'Fail: {str(e)}',
                'correct_indentation': 'Fail: Skipped',
                'no_deprecations': 'Fail: Skipped',
                'correct_functionality': 'Fail: Skipped',
                'functions_retrieved': 0,
                'functions_with_context': 0
            })
            continue

        generated_code = modernized_code
        print(generated_code)        
        if not generated_code:
            print("RAG did not generate code. Skipping.")
            functions_retrieved = len(retrieved_context) if isinstance(retrieved_context, dict) else 0
            functions_with_context = sum(1 for v in retrieved_context.values() if v) if isinstance(retrieved_context, dict) else 0
            results_list.append({
                'sample_index': i, 
                'compiles': 'Fail: No code generated',
                'correct_indentation': 'Fail: Skipped',
                'no_deprecations': 'Fail: Skipped',
                'correct_functionality': 'Fail: Skipped',
                'functions_retrieved': functions_retrieved,
                'functions_with_context': functions_with_context
            })
            continue

        dedented_code = textwrap.dedent(generated_code)
        indented_code = textwrap.indent(dedented_code, "    ")
        full_code = sample['code_before'] + indented_code + sample['code_after']
        full_code_ni = sample['code_before'] + generated_code + sample['code_after']
        function_name = sample['code_before'].split('def ')[1].split('(')[0].strip()
        
        functions_retrieved = len(retrieved_context) if isinstance(retrieved_context, dict) else 0
        functions_with_context = sum(1 for v in retrieved_context.values() if v) if isinstance(retrieved_context, dict) else 0
        
        #COMPILATION CHECK
        compiles, compiles_msg, compiled_function = check_compiles(function_name, full_code)
        #INDENTATION CHECK  
        indentation, indentation_msg = (check_indentation(function_name, full_code_ni) if compiles else (False, "Skipped"))
        #DEPRECATION CHECK
        test_input = eval(sample['test_cases'][0]['input'], EVAL_GLOBALS)
        no_deprecations, no_deprecations_msg = (check_no_deprecations(compiled_function, test_input) if compiles else (False, "Skipped"))
        #FUNCTIONALITY CHECK
        functionality, functionality_msg = (check_functionality(compiled_function, sample['test_cases']) if compiles and no_deprecations else (False, "Skipped"))

        results_list.append({
            'sample_index': i,
            'compiles': 'Pass' if compiles else f'Fail: {compiles_msg}',
            'correct_indentation': 'Pass' if indentation else f'Fail: {indentation_msg}',
            'no_deprecations': 'Pass' if no_deprecations else f'Fail: {no_deprecations_msg}',
            'correct_functionality': 'Pass' if functionality else f'Fail: {functionality_msg}',
            'functions_retrieved': functions_retrieved,
            'functions_with_context': functions_with_context
        })

    print("\nEvaluation DONE")

    detailed_df = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(DETAILED_RESULTS_CSV), exist_ok=True)
    detailed_df.to_csv(DETAILED_RESULTS_CSV, index=False)

    metrics = {}
    metrics['total_samples'] = total_samples
    metrics['compiles'] = detailed_df['compiles'].str.startswith('Pass').sum()
    metrics['correct_indentation'] = detailed_df['correct_indentation'].str.startswith('Pass').sum()
    metrics['no_deprecations'] = detailed_df['no_deprecations'].str.startswith('Pass').sum()
    metrics['correct_functionality'] = detailed_df['correct_functionality'].str.startswith('Pass').sum()
    
    metrics['avg_functions_retrieved'] = detailed_df['functions_retrieved'].mean()
    metrics['avg_functions_with_context'] = detailed_df['functions_with_context'].mean()

    summary_score = (
        metrics['compiles'] +
        metrics['correct_indentation'] +
        metrics['no_deprecations'] +
        (3 * metrics['correct_functionality'])
    )
    metrics['summary_score'] = summary_score

    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(SUMMARY_METRICS_CSV, index=False)
    print("\nSummary Metrics:")
    print(summary_df.to_string())
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
