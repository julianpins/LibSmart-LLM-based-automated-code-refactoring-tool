import json
import warnings
import numpy as np
from typing import Callable, Any
import numpy as np
import pickle
import os
import ast

import numpy as np
from numpy import ma
import numpy_financial as npf


PATH_TO_VALIDATION = 'data/datasets/validation_data.json'

# SCORING
POINTS_COMPILE = 1
POINTS_NO_DEPRECATION = 2
POINTS_PER_TEST_CASE = 1
POINTS_CODE_MATCH = 0  # Optional

EVAL_GLOBALS = {
    'np': np,
    'ma': ma,
    'os': os,
    'ast': ast,
    'pickle': pickle,
    'npf': npf,
}

# Mock LLM function
def get_llm_suggestion(input_code_block, all_samples, index):
    print("Getting the models suggested code ...")
    suggested_code = all_samples[index]['output']
    print(f"Model Suggestion:\n```python\n{suggested_code.strip()}\n```")
    return suggested_code


#executes function with sample input and checks for deprecation warnings
def has_deprecation(f, sample_input):
    if not callable(f):
        return None

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            if isinstance(sample_input, tuple):
                f(*sample_input)
            else:
                f(sample_input)
        except AttributeError as e:
            if "module 'numpy' has no attribute" in str(e):
                print(f"Deprecation found: The function uses a removed attribute -> {e}")
                return True
            print(f"The function failed with an unrelated attribute error: {e}")
            return None 
        except Exception as e:
            print(f"The function failed with an unexpected error: {e}")
            return None 

        for warning in captured_warnings:
            if issubclass(warning.category, DeprecationWarning):
                print(f"Deprecation found: {warning.message}")
                return True
        if captured_warnings:
            return None
    return False

# compare outputs for arbitrary (numpy) datatypes
def compare_outputs(actual, expected):
    if isinstance(actual, np.ma.MaskedArray) and actual.ndim == 0:
        actual = actual.item() if not actual.mask else np.ma.masked
    if isinstance(expected, np.ma.MaskedArray) and expected.ndim == 0:
        expected = expected.item() if not expected.mask else np.ma.masked
    if actual is np.ma.masked and expected is np.ma.masked:
        return True
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False
        return all(compare_outputs(a, e) for a, e in zip(actual, expected))
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.replace(" ", "").replace("\n", "") == expected.replace(" ", "").replace("\n", "")
    actual_array = isinstance(actual, (np.ndarray, list, tuple))
    expected_array = isinstance(expected, (np.ndarray, list, tuple))
    if actual_array and expected_array:
        try:
            actual_arr = np.asarray(actual)
            expected_arr = np.asarray(expected)
            if np.issubdtype(actual_arr.dtype, np.floating) or \
               np.issubdtype(expected_arr.dtype, np.floating) or \
               np.issubdtype(actual_arr.dtype, np.complexfloating) or \
               np.issubdtype(expected_arr.dtype, np.complexfloating):
                return np.allclose(actual_arr, expected_arr, equal_nan=True)
            return np.array_equal(actual_arr, expected_arr)
        except (ValueError, TypeError):
            return False
    if actual_array != expected_array:
        return False
    return actual == expected

#calculates score for all samples in validation_data
def run_test_suite():
    try:
        with open(PATH_TO_VALIDATION, 'r') as f:
            test_samples = json.load(f)
    except FileNotFoundError:
        print("ERROR: Validation data not found.")
        return
    
    total_score = 0
    max_possible_score = 0


    print(f"Starting Test Suite for {len(test_samples)} samples:")

    for i, sample in enumerate(test_samples):
        sample_score = 0
        max_sample_score = POINTS_COMPILE + POINTS_NO_DEPRECATION + (POINTS_PER_TEST_CASE * len(sample['test_cases'])) + POINTS_CODE_MATCH
        max_possible_score += max_sample_score
        
        print(f"\n{'='*42}")
        print(f"Running Test {i+1}/{len(test_samples)}")
        #print(f"Input Code:\n```python\n{sample['input'].strip()}\n```")

        suggested_code = get_llm_suggestion(sample['input'], test_samples, i)

        full_code = (
            sample['code_before'] + "\n" +
            suggested_code + "\n" +
            sample['code_after']
        )
        
        # COMPILATION
        function_name = sample['code_before'].split('def ')[1].split('(')[0].strip()

        execution_scope = {}
        try:
            exec(full_code, EVAL_GLOBALS, execution_scope)
            compiled_function = execution_scope.get(function_name)
            if compiled_function:
                print(f"✅ COMPILE CHECK: Success (+{POINTS_COMPILE} pts)")
                sample_score += POINTS_COMPILE
            else:
                raise SyntaxError("Function definition not found in execution scope.")
        except Exception as e:
            print(f"❌ COMPILE CHECK: Failed. Error: {e}")
            print(f"Sample {i+1} total score: {sample_score}/{max_sample_score}")
            total_score += sample_score
            continue

        # DEPRECATION REMOVAL
        test_input = eval(sample['test_cases'][0]['input'], EVAL_GLOBALS)
        deprecation = has_deprecation(compiled_function, test_input)
        if deprecation is False:
            print(f"✅ DEPRECATION CHECK: Success (+{POINTS_NO_DEPRECATION} pts)")
            sample_score += POINTS_NO_DEPRECATION
        elif deprecation is None:
            print(f"❌ Function crashed during deprecation check. Stopping test")
            print(f"Sample {i+1} total score: {sample_score}/{max_sample_score}")
            total_score += sample_score
            continue
        else:
            print(f"❌ DEPRECATION CHECK: Failed.")
            print(f"Sample {i+1} total score: {sample_score}/{max_sample_score}")
            total_score += sample_score
            continue

        # FUNCTIONAL CORRECTNESS
        print("--- Running I/O Test Cases ---")
        all_tests_passed = True
        test_cases_passed_count = 0
        for j, test_case in enumerate(sample['test_cases']):
            try:
                eval_env = {'np': np, 'ma': ma}
                test_input = eval(test_case['input'], eval_env)
                expected_output = eval(test_case['expected_output'], eval_env)

                if isinstance(test_input, tuple):
                    actual_output = compiled_function(*test_input)
                else:
                    actual_output = compiled_function(test_input)
                
                if compare_outputs(actual_output, expected_output):
                    print(f"  ✅ Test Case {j+1}: Passed")
                    test_cases_passed_count += 1
                else:
                    print(f"  ❌ Test Case {j+1}: Failed.")
                    print(f"     Input:    {test_case['input']}")
                    print(f"     Expected: {repr(expected_output)}")
                    print(f"     Got:      {repr(actual_output)}")
                    all_tests_passed = False
            except Exception as e:
                print(f"  ❌ Test Case {j+1}: Error during execution: {e}")
                all_tests_passed = False
        
        case_points = test_cases_passed_count * POINTS_PER_TEST_CASE
        sample_score += case_points
        print(f"I/O CHECK Result: {test_cases_passed_count}/{len(sample['test_cases'])} passed (+{case_points} pts)")
        
        if not all_tests_passed:
            print(f"Sample {i+1} total score: {sample_score}/{max_sample_score}")
            total_score += sample_score
            continue

        #CHECK 4: CODE MATCH
        '''if suggested_code.strip() == sample['output'].strip():
            print(f"✅ CODE MATCH CHECK: Success (+{POINTS_CODE_MATCH} pts)")
            sample_score += POINTS_CODE_MATCH
        else:
            print(f"ℹ️  CODE MATCH CHECK: Failed (functionally correct but not an exact match).")
        '''

        print(f"--- Sample {i+1} total score: {sample_score}/{max_sample_score} ---")
        total_score += sample_score

    # --- FINAL REPORT ---
    print(f"\n{'='*50}")
    print("COMPLETE")
    performance_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    print(f"Total Score: {total_score} / {max_possible_score}")
    print(f"Overall Performance: {performance_percentage:.2f}%")
    print('='*50)

if __name__ == '__main__':
    run_test_suite()
    