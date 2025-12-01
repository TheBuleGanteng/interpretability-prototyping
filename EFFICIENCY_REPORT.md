# Efficiency Analysis Report

## Overview

This report documents efficiency improvements identified in the interpretability-prototyping codebase. The analysis covers all three phase notebooks and identifies patterns that could significantly improve performance.

## Issues Identified

### 1. Sequential Text Processing Instead of Batching (High Impact)

**Location:** Multiple files

**Problem:** Throughout the codebase, texts are processed one at a time in loops rather than being batched together. This is highly inefficient because:
- Each call to `model.run_with_cache()` has overhead
- GPU/CPU parallelism is not utilized
- The model can process multiple texts simultaneously

**Affected Code:**

In `phase_1_setup_intro.ipynb` (approximate lines 162-179):
```python
for text in texts:
    logits, cache = model.run_with_cache(text)
    acts = cache[hook_name][0, -1, :]
    # ... process one text at a time
```

In `phase_2_feature_exploration.ipynb` (approximate lines 162-178, 337-343):
```python
for text in texts:
    logits, cache = model.run_with_cache(text)
    acts = cache[hook_name][0, -1, :]
    # ... process one text at a time
```

In `phase_2_feature_exploration.ipynb` - `explore_feature()` function (approximate lines 738-748):
```python
for text in texts_to_test:
    logits, cache = model.run_with_cache(text)
    # ... process one text at a time
```

In `phase_2_feature_exploration.ipynb` - `find_features_for_concept()` function (approximate lines 831-845):
```python
for text in concept_texts:
    logits, cache = model.run_with_cache(text)
    # ... process one text at a time
```

**Solution:** Batch all texts together and process them in a single forward pass:
```python
tokens = model.to_tokens(texts, prepend_bos=True)
seq_lens = (tokens != model.tokenizer.pad_token_id).sum(dim=1)  # Get length of each sequence (excluding padding)
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
activations = torch.stack([
    cache[hook_name][i, seq_lens[i]-1, :] for i in range(tokens.shape[0])
])  # Get actual last token for each text
features = sae.encode(activations)
**Estimated Impact:** 5-10x speedup for feature extraction operations.

---

### 2. Redundant Feature Extraction in display_comparison_table (High Impact)

**Location:** `phase_3_SAE_comparison.ipynb`, approximate lines 762-852

**Problem:** The `display_comparison_table()` function re-extracts features for each SAE three separate times (once for strongest features, once for frequent features, once for selective features). This triples the computation time unnecessarily.

**Affected Code:**
```python
# Row 1: Top 5 Strongest Features - extracts features
features_tensor = extract_features(texts, sae_obj, hook_point)

# Row 2: Top 5 Most Frequent Features - extracts features AGAIN
features_tensor = extract_features(texts, sae_obj, hook_point)

# Row 3: Top 5 Most Selective Features - extracts features AGAIN
features_tensor = extract_features(texts, sae_obj, hook_point)
```

**Solution:** Extract features once per SAE and reuse:
```python
# Extract once at the start
features_cache = {}
for sae_name in sae_names:
    sae_obj = loaded_saes[sae_name]['sae']
    hook_point = loaded_saes[sae_name]['config']['hook_point']
    features_cache[sae_name] = extract_features(texts, sae_obj, hook_point)

# Then use features_cache[sae_name] throughout the function
```

**Estimated Impact:** 3x speedup for the comparison table generation.

---

### 3. Inefficient Category Index Lookup (Medium Impact)

**Location:** `phase_3_SAE_comparison.ipynb`, `analyze_specialists()` function (approximate lines 490-493)

**Problem:** For each category, the function iterates through all texts to find which indices belong to that category using a string comparison:
```python
indices = [i for i, text in enumerate(texts) if text in cat_texts]
```

This is O(n*m) where n is the number of texts and m is the number of texts in the category.

**Solution:** Precompute category indices once and pass them to the function, or use a dictionary mapping texts to indices:
```python
# Precompute once
text_to_idx = {text: i for i, text in enumerate(texts)}
# Then use: indices = [text_to_idx[text] for text in cat_texts]
```

**Estimated Impact:** Minor speedup, but cleaner code.

---

### 4. Duplicate Function Definitions (Low Impact, Code Quality)

**Location:** `phase_3_SAE_comparison.ipynb`

**Problem:** The `neuronpedia_link()` function is defined twice:
- Approximate lines 559-560 (module level)
- Approximate lines 681-682 (inside `display_comparison_table()`)

**Solution:** Remove the duplicate definition inside `display_comparison_table()` and use the module-level function.

---

### 5. Inefficient Cache Checking with os.walk (Low Impact)

**Location:** `phase_3_SAE_comparison.ipynb`, `check_sae_cached()` function (approximate lines 122-135)

**Problem:** Uses `os.walk()` to traverse the entire HuggingFace cache directory, which can be slow if the cache is large.

**Solution:** Use more targeted path checking:
```python
def check_sae_cached(sae_path):
    direct_path = sae_lens_cache / sae_path
    if direct_path.exists() and (direct_path / "sae_weights.safetensors").exists():
        return True, direct_path
    # Use glob with specific pattern instead of os.walk
    matches = list(hf_cache.glob(f"**/{sae_path}/sae_weights.safetensors"))
    if matches:
        return True, matches[0].parent
    return False, None
```

---

### 6. Missing torch.no_grad() Context (Low Impact)

**Location:** Various places in phase_1 and phase_2

**Problem:** Some tensor operations that don't require gradients are not wrapped in `torch.no_grad()`, which wastes memory tracking gradients.

**Example in `phase_1_setup_intro.ipynb` (approximate lines 162-179):**
```python
for text in texts:
    logits, cache = model.run_with_cache(text)  # Should be in no_grad context
```

**Solution:** Wrap inference code in `torch.no_grad()`:
```python
with torch.no_grad():
    for text in texts:
        logits, cache = model.run_with_cache(text)
```

---

## Recommendations

### Priority 1 (High Impact)
1. **Batch text processing** - Convert all sequential text processing loops to batched operations
2. **Cache extracted features** - Extract features once per SAE and reuse throughout analysis

### Priority 2 (Medium Impact)
3. **Precompute category indices** - Avoid repeated string lookups in analyze_specialists

### Priority 3 (Code Quality)
4. **Remove duplicate function definitions**
5. **Optimize cache checking**
6. **Add torch.no_grad() contexts consistently**

## Implementation Plan

This PR implements fix #1 (batch text processing) in `phase_2_feature_exploration.ipynb` as it provides the highest impact improvement. The change converts the sequential text processing loop in the diverse dataset feature extraction to use batched processing.
