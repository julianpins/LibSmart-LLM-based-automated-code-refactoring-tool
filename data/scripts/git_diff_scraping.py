import os
import tempfile
from git import Repo
from tqdm import tqdm
from collections import Counter

#keywords to search in commits
KEYWORDS = ["deprecat", "refactor", "replace"]

TARGET_FUNCTIONS  = [
    ".T",
    "np.asscalar",           "numpy.asscalar",
    "np.alen",               "numpy.alen",
    "np.rank",               "numpy.rank",
    "np.msort",              "numpy.msort",
    "np.product",            "numpy.product",
    "np.fastCopyAndTranspose","numpy.fastCopyAndTranspose",
    "np.bool",    "numpy.bool",
    "np.bytes",   "numpy.bytes",
    "np.complex", "numpy.complex",
    "np.float",   "numpy.float",
    "np.int",     "numpy.int",
    "np.object",  "numpy.object",
    "np.str",     "numpy.str",
    "np.str0",    "numpy.str0",
    "np.bytes0",  "numpy.bytes0",
    "np.void0",   "numpy.void0",
    "np.fv",   "numpy.fv",
    "np.pv",   "numpy.pv",
    "np.npv",  "numpy.npv",
    "np.rate", "numpy.rate",
    "np.nper", "numpy.nper",
    "np.pmt",  "numpy.pmt",
    "np.ppmt", "numpy.ppmt",
    "np.irr",  "numpy.irr",
    "np.mirr", "numpy.mirr",
    "np.ipmt", "numpy.ipmt",
    "np.chararray",              
    "numpy.chararray", 
    "np.core.umath_tests", 
    "np.set_string_function",
    "np.deprecate", "numpy.deprecate",
    "np.deprecate_with_doc", "numpy.deprecate_with_doc",
    "np.UPDATEIFCOPY", "numpy.UPDATEIFCOPY"
]
function_counter = Counter()

MAX_DATA_PER_FUNCTION = 5

TARGET_LIBRARIES = ["numpy", "np."]

REPOS = [
    "https://github.com/CALFEM/calfem-python.git"
    "https://github.com/inkstitch/colormath.git"
    "https://github.com/scikit-tda/scikit-hubness.git"
    
    "https://github.com/numpy/numpy.git"
    "https://github.com/scipy/scipy.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    #"https://github.com/astropy/astropy.git"
    #"https://github.com/scikit-image/scikit-image.git"
    #"https://github.com/statsmodels/statsmodels.git"
    #"https://github.com/xarray/xarray.git"
]

#check if diff includes numpy function
def is_relevant_diff(diff_text):
    return any(lib in diff_text for lib in TARGET_LIBRARIES)

#checks if relevant function contained in input_code
def contains_func(diff_text):
    found_funcs = [func for func in TARGET_FUNCTIONS if func in diff_text]
    if not found_funcs:
        return False
    if all(function_counter[func] >= MAX_DATA_PER_FUNCTION for func in found_funcs):
        return False
    for func in found_funcs:
        function_counter[func] += 1
    return True

#sort diff for old and new code by searching for - and + at beginning of lines
def extract_diffs_from_commit(commit):
    diffs = []
    for diff in commit.diff(commit.parents[0], create_patch=True):
        if diff.a_path and diff.a_path.endswith(".py") and diff.diff:
            old_lines = []
            new_lines = []
            for line in diff.diff.decode(errors="ignore").split("\n"):
                if line.startswith("-") and not line.startswith("---"):
                    old_lines.append(line[1:])
                elif line.startswith("+") and not line.startswith("+++"):
                    new_lines.append(line[1:])
            if old_lines and new_lines:
                diffs.append({
                    "old": "\n".join(old_lines),
                    "new": "\n".join(new_lines)
                })
    return diffs

#process entire repository
#for every commit check if one of the keywords appears in commit message
#if diff is relevant, build json
def process_repo(repo_url):
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {repo_url} ...")
        repo_path = os.path.join(tmpdir, os.path.basename(repo_url))
        repo = Repo.clone_from(repo_url, repo_path)
        print("Analyzing commits ...")
        for commit in tqdm(list(repo.iter_commits())):
            msg = commit.message.lower()
            if any(keyword in msg for keyword in KEYWORDS):
                try:
                    diffs = extract_diffs_from_commit(commit)
                    for d in diffs:
                        if is_relevant_diff(d["old"] + d["new"]):
                            if contains_func(d["old"]):
                                results.append({
                                    "input_code": d["old"],
                                    "package": "numpy",
                                    "context": commit.message.strip(),
                                    "modern_code": d["new"]
                                })

                except Exception as e:
                    print(f"Error in commit {commit.hexsha}: {e}")
        repo.close()
    return results


#iterate over all selected repos
all_diffs = []
for repo_url in REPOS:
    diffs = process_repo(repo_url)
    all_diffs.extend(diffs)


# Save results
import json
with open("numpy_refactor_data.json", "w") as f:
    json.dump(all_diffs, f, indent=2)

print(f"Extracted {len(all_diffs)} deprecated/refactored pairs.")