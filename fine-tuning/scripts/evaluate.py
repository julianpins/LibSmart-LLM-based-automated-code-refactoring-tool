#Uncomment:
#!pip install -q -U "torch==2.3.1" "transformers==4.41.2" "peft==0.11.1" "accelerate==0.30.1" "trl==0.9.4" "datasets==2.19.2" "bitsandbytes==0.43.1" numpy_financial

import json
import warnings
import textwrap
import re
import os
import ast
import pickle

#Uncomment:
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
import numpy as np
from numpy import ma
import numpy_financial as npf

#For different models
#"google/gemma-2b", "codellama/CodeLlama-7b-hf", "mistralai/Mistral-7B-Instruct-v0.2"
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

#depending on model
ADAPTER_PATH = "./mistral-7b-libsmart"

PATH_TO_VALIDATION = 'data/datasets/validation_data.json'

# Scoring
POINTS_COMPILE = 1
POINTS_CORRECT_INDENTATION = 1
POINTS_NO_DEPRECATION = 2
POINTS_PER_TEST_CASE = 2
POINTS_CODE_MATCH = 0 # optional

EVAL_GLOBALS = {
    'np': np,
    'ma': ma,
    'os': os,
    'ast': ast,
    'pickle': pickle,
    'npf': npf,
}

def load_model_and_tokenizer(base_model_id, adapter_path):
    print(f"Loading base model: {base_model_id} and adapter from: {adapter_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token    
    model = PeftModel.from_pretrained(model, adapter_path)    
    return model, tokenizer

def get_model_suggestion(model, tokenizer, input_code, all_samples, index):
    #MOCK:
    print("Getting suggested code: Using expected output")
    return all_samples[index]['output']

    #Uncomment for actual model-response
    '''
    instruction = "Analyze the following Python code for a deprecated NumPy function and generate a JSON object with the suggested fix and context."
    prompt = f"<s>[INST] {instruction}\n\n### INPUT CODE:\n```python\n{input_code}\n``` [/INST]\n{output_json}</s>"
    # For gemma
    # prompt = f"<start_of_turn>user\n{instruction}\n\n### INPUT CODE:\n```python\n{input_code}\n```<end_of_turn>\n<start_of_turn>model\n{output_json}<end_of_turn>"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

    #Generate response
    outputs = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"Model raw output:\n{response_text}")
    #find JSON object within the raw text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_string = json_match.group(0)
        try:
            parsed_json = json.loads(json_string)
            suggested_code = parsed_json.get("output", "")
            print(f"Parsed suggestion:\n```python\n{suggested_code.strip()}\n```")
            return suggested_code
        except json.JSONDecodeError:
            print("Warning: Couldn't parse JSON. Using raw output as fallback.")
            return response_text
    else:
        print("Warning: Couldn't find a valid JSON block in output.")
        return ""
    '''

#checks for deprecation errors / warnings
def has_deprecation(f, sample_input):
    if not callable(f): return None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if isinstance(sample_input, tuple): f(*sample_input)
            else: f(sample_input)
        except AttributeError as e:
            if "module 'numpy' has no attribute" in str(e): return True
            return None
        except Exception: return None
        for item in w:
            if issubclass(item.category, DeprecationWarning): return True
    return False


#cover all important datatypes in numpy
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


#calculates score for all samples in validation_data
def run_test_suite():
    #Uncomment:
    #model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID, ADAPTER_PATH)
    model, tokenizer = None, None
    try:
        with open(PATH_TO_VALIDATION, 'r') as f:
            test_samples = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Validation data not found at '{PATH_TO_VALIDATION}'.")
        return
    
    total_score, max_possible_score = 0, 0
    print(f"\nStarting Test Suite for {len(test_samples)} samples:")

    for i, sample in enumerate(test_samples):
        sample_score = 0
        max_sample_score = POINTS_COMPILE + POINTS_CORRECT_INDENTATION + POINTS_NO_DEPRECATION + (POINTS_PER_TEST_CASE * len(sample['test_cases']))
        max_possible_score += max_sample_score
        print(f"\n{'='*42}")
        print(f"Running Test {i+1}/{len(test_samples)}")

        #Get model suggestion
        suggested_code = get_model_suggestion(model, tokenizer, sample['input'], test_samples, i)
        if not suggested_code:
            print("Model returned an empty suggestion. Skipping sample.")
            continue
        
        #COMPILATION CHECK
        full_code = sample['code_before'] + "\n" + suggested_code + "\n" + sample['code_after']
        dedented_code = textwrap.dedent(full_code).strip()
        indented_code = textwrap.indent(dedented_code, "")

        function_name = sample['code_before'].split('def ')[1].split('(')[0].strip()
        execution_scope, compiled_function = {}, None
        try:
            exec(indented_code, EVAL_GLOBALS, execution_scope)
            compiled_function = execution_scope.get(function_name)
            if not callable(compiled_function):
                raise NameError(f"Function '{function_name}' was not defined correctly.")
            print(f"✅ COMPILE CHECK: Success (+{POINTS_COMPILE} pts)")
            sample_score += POINTS_COMPILE
        except Exception as e:
            print(f"❌ COMPILE CHECK: Failed. Error: {e}")
            total_score += sample_score
            continue

        #INDENTATION
        try:
            exec(full_code, EVAL_GLOBALS, execution_scope)
            compiled_function = execution_scope.get(function_name)
            print(f"✅ INDENTATION CHECK: Success (+{POINTS_CORRECT_INDENTATION} pts)")
            sample_score += POINTS_CORRECT_INDENTATION
        except Exception as e:
            print(f"❌ INDENTATION CHECK: Failed. Error: {e}")

        #DEPRECATION CHECK
        try:
            test_input = eval(sample['test_cases'][0]['input'], EVAL_GLOBALS)
            deprecation_found = has_deprecation(compiled_function, test_input)
            if deprecation_found is False:
                print(f"✅ DEPRECATION CHECK: Success (+{POINTS_NO_DEPRECATION} pts)")
                sample_score += POINTS_NO_DEPRECATION
            elif deprecation_found is True:
                print("❌ DEPRECATION CHECK: Failed. Deprecation found.")
                total_score += sample_score; continue
            else:
                print("❌ DEPRECATION CHECK: Function crashed during test.")
                total_score += sample_score; continue
        except Exception as e:
            print(f"❌ DEPRECATION CHECK: Could not run due to error in test case: {e}")
            total_score += sample_score; continue
            
        #FUNCTIONAL CORRECTNESS
        print("Running I/O Test Cases")
        passed_count = 0
        for j, test_case in enumerate(sample['test_cases']):
            try:
                test_input = eval(test_case['input'], EVAL_GLOBALS)
                expected_output = eval(test_case['expected_output'], EVAL_GLOBALS)
                actual_output = compiled_function(*test_input) if isinstance(test_input, tuple) else compiled_function(test_input)
                if compare_outputs(actual_output, expected_output):
                    print(f"  ✅ Test Case {j+1}: Passed")
                    passed_count += 1
                else:
                    print(f"  ❌ Test Case {j+1}: Failed. [Expected: {repr(expected_output)}, Got: {repr(actual_output)}]")
            except Exception as e:
                print(f"  ❌ Test Case {j+1}: Error during execution: {e}")
        
        case_points = passed_count * POINTS_PER_TEST_CASE
        sample_score += case_points
        print(f"I/O CHECK Result: {passed_count}/{len(sample['test_cases'])} passed (+{case_points} pts)")
        
        print(f"--- Sample {i+1} total score: {sample_score}/{max_sample_score} ---")
        total_score += sample_score

    #Final Score
    print(f"\n{'='*42}")
    print("COMPLETE")
    print(f"Total Score: {total_score} / {max_possible_score}")
    performance = (total_score / max_possible_score) * 100
    print(f"Overall Performance: {performance:.2f}%")
    print('='*42)

if __name__ == '__main__':
    run_test_suite()