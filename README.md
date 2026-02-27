# LLM Interpretability Prototyping

## Project Purpose

This project is an educational exploration of Large Language Model (LLM) interpretability techniques, specifically focusing on **Sparse Autoencoders (SAEs)** as demonstrated in Anthropic's research: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

### Core Concept

Sparse Autoencoders help us understand what's happening inside neural networks by:
- **Disaggregating** dense, polysemantic activations (where each dimension represents multiple concepts)
- **Into** sparse, monosemantic features (where each feature represents a single interpretable concept)

This transformation makes it possible to see what an LLM is "thinking about" when processing text.

## Project Goals

1. Build hands-on understanding of SAE-based interpretability
2. Visualize how LLM internal representations can be decomposed into interpretable features
3. Explore which features activate for different inputs
4. Test whether specialist features detect semantic meaning or surface syntax across model scales
5. Investigate the causal role of specialist features via ablation experiments

---

## Getting Started

### Prerequisites

**Phases 0–2 (local Jupyter notebooks):**
- Python 3.9+
- 8GB+ RAM
- Internet connection for initial model/SAE downloads

**Phase 3 (Google Colab):**
- Google account with access to Google Colab
- Colab Pro recommended (GPU runtime required for Gemma 2 9B)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd interpretability-prototyping
```

### 2. Create and Activate Virtual Environment (Phases 0–2)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/Mac/WSL)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies (Phases 0–2)
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformer-lens sae-lens
pip install transformers datasets einops
pip install plotly jupyter ipywidgets
pip install numpy pandas matplotlib
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained SAEs (Phases 0–2, Optional but Recommended)
Download SAEs via terminal to avoid Jupyter progress bar issues:
```bash
# Download Layer 6 SAE
python3 -c "from sae_lens import SAE; SAE.from_pretrained('gpt2-small-res-jb', 'blocks.6.hook_resid_pre', 'cpu'); print('✅ Layer 6 downloaded')"

# Download additional layers as needed (8, 10, 11)
python3 -c "from sae_lens import SAE; SAE.from_pretrained('gpt2-small-res-jb', 'blocks.8.hook_resid_pre', 'cpu'); print('✅ Layer 8 downloaded')"
python3 -c "from sae_lens import SAE; SAE.from_pretrained('gpt2-small-res-jb', 'blocks.10.hook_resid_pre', 'cpu'); print('✅ Layer 10 downloaded')"
python3 -c "from sae_lens import SAE; SAE.from_pretrained('gpt2-small-res-jb', 'blocks.11.hook_resid_pre', 'cpu'); print('✅ Layer 11 downloaded')"
```

Move downloaded SAEs to a consistent cache location:
```bash
# Create cache directory if needed
mkdir -p ~/.cache/sae_lens

# Copy SAEs (use -rL to dereference symlinks)
cp -rL ~/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/blocks.*.hook_resid_pre ~/.cache/sae_lens/
```

### 5. Launch Jupyter and Run Notebooks (Phases 0–2)

**Option A: Command Line**
```bash
jupyter notebook
```
Then navigate to `notebooks/` and open the desired notebook.

**Option B: VSCode**
1. Open the project folder in VSCode
2. Open any `.ipynb` file in the `notebooks/` folder
3. Select the `venv` Python interpreter (bottom right of VSCode)
4. Run cells with Shift+Enter

### 6. Run Notebooks / Colab in Order

The analyses are designed to be run sequentially:

| Phase | Notebook / Colab | Description |
|-------|------------------|-------------|
| 0a | `notebooks/phase_0a.ipynb` | Initial setup, model loading, basic SAE exploration |
| 0b | `notebooks/phase_0b_feature_exploration.ipynb` | Feature analysis, heatmaps, specialist search |
| 1 | `notebooks/phase_1_SAE_comparison.ipynb` | Multi-SAE comparison across layers |
| 2 | `notebooks/phase_2_semantics_vs_ideas.ipynb` | Feature activation: syntax vs. meaning (GPT-2 Small) |
| 3 | [Phase 3 Colab](https://colab.research.google.com/drive/11dWRTZ9Jqf5-UQU1-Zw2osxiEF1Sn9Jh?usp=sharing) | Cross-model comparison: GPT-2 Small vs. Gemma 2 9B |

### Troubleshooting (Phases 0–2)

**Model/SAE download stalls in Jupyter:**
- Use terminal downloads as shown in Step 4 above
- Then use `SAE.load_from_disk(path)` in notebooks

**Import errors:**
- Verify virtual environment is activated: `which python` should show `venv/bin/python`
- Reinstall dependencies: `pip install -r requirements.txt`

**SAE loading hangs:**
- Check if files exist: `ls ~/.cache/sae_lens/`
- Ensure files are actual files, not broken symlinks: `ls -la ~/.cache/sae_lens/blocks.6.hook_resid_pre/`

---

## Published Articles

| Phase | Substack | LessWrong |
|-------|----------|-----------|
| Phase 1 | [A Peek Inside the Black Box: Part 1](https://matthewmcdonnell.substack.com/p/a-peek-inside-the-black-box-part?r=b2gju) | [A Black Box Made Less Opaque: Part 1](https://www.lesswrong.com/posts/QRM3q9ZhLDZuxuDbz/a-black-box-made-less-opaque-part-1) |
| Phase 2 | [A Peek Inside the Black Box: Part 2](https://matthewmcdonnell.substack.com/p/a-peek-inside-the-black-box-part-2?r=b2gju) | [A Black Box Made Less Opaque: Part 2](https://www.lesswrong.com/posts/Qnm6gAFnCPaJsbhSS/a-black-box-made-less-opaque-part-2) |
| Phase 3 | *(forthcoming)* | *(forthcoming)* |

---

## Current Status

✅ **Phase 0a: Initial Setup & Exploration** — COMPLETED (2025-10-23)
✅ **Phase 0b: Feature Exploration & Analysis** — COMPLETED (2025-11-05)
✅ **Phase 1: SAE Comparison & Advanced Analysis** — COMPLETED (2025-12-02)
✅ **Phase 2: Semantics vs. Syntax Analysis** — COMPLETED (2025-12-19)
✅ **Phase 3: Cross-Model Comparison (GPT-2 Small vs. Gemma 2 9B)** — COMPLETED (2026-02-27)

---

## Roadmap

### Phase 0a: Environment Setup & Exploration ✅ COMPLETED

Initial setup phase establishing the technical foundation: loaded GPT-2 Small via TransformerLens, integrated pre-trained SAEs via SAELens, and verified the interpretability pipeline. Achieved 87.35% sparsity (3,108/24,576 features active) and investigated Feature #10399 using Neuronpedia to understand feature interpretation methodology.

---

### Phase 0b: Feature Exploration & Analysis ✅ COMPLETED

Systematic feature exploration phase: built a 70-text diverse dataset across 7 categories (Python, URLs, Math, Non-English, Social/Emoji, Formal, Conversational), implemented multi-criteria feature discovery (strongest, frequent, selective, category-specific), and built the specialist search framework. Found initial Math specialist (Feature #18522, score +6) and established activation strength weighting methodology.

---

### Phase 1: SAE Comparison & Advanced Analysis ✅ COMPLETED
**Goal:** Compare specialist feature emergence across multiple SAE layers

**Corresponding Article:** "A Peek Inside the Black Box: Part 1" — [Substack](https://matthewmcdonnell.substack.com/p/a-peek-inside-the-black-box-part?r=b2gju) · [LessWrong](https://www.lesswrong.com/posts/QRM3q9ZhLDZuxuDbz/a-black-box-made-less-opaque-part-1)

**Key Findings:**

1. **Layer Depth Correlates with Specialization:**
   - Layer 6: 5/7 category specialists found
   - Layer 8: 6/7 specialists found
   - Layer 10: 7/7 specialists found (complete coverage)
   - Layer 11: 7/7 specialists found (complete coverage)

2. **Category Difficulty Hierarchy:**
   - Easy to specialize (found at layer 6): Math, Non-English, URLs, Python, Formal
   - Hard to specialize (only layers 10-11): Social, Conversational

3. **Critical Methodological Discovery:**
   - Padding token masking bug initially suppressed specialist detection entirely
   - Fix improved results from 0-1 specialists per SAE to 5-7 specialists
   - Lesson: Negative results require careful methodological verification

4. **Architecture Insight:**
   - Residual stream SAEs ARE viable for finding domain specialists
   - Previous hypothesis that residual streams favor general features was incorrect—the issue was methodological, not architectural

**SAEs Tested:**
- `blocks.6.hook_resid_pre` (Layer 6 Residual Stream)
- `blocks.8.hook_resid_pre` (Layer 8 Residual Stream)
- `blocks.10.hook_resid_pre` (Layer 10 Residual Stream)
- `blocks.11.hook_resid_pre` (Layer 11 Residual Stream)

---

### Phase 2: Semantics vs. Syntax Analysis ✅ COMPLETED
**Goal:** Test whether specialist features detect semantic meaning or surface-level syntactic patterns

**Corresponding Article:** "A Peek Inside the Black Box: Part 2" — [Substack](https://matthewmcdonnell.substack.com/p/a-peek-inside-the-black-box-part-2?r=b2gju) · [LessWrong](https://www.lesswrong.com/posts/Qnm6gAFnCPaJsbhSS/a-black-box-made-less-opaque-part-2)

**Research Question:** Do SAE specialist features respond to the underlying *concept* (semantic meaning) or merely the *surface form* (syntactic patterns like operators and symbols)?

**Methodology:**
- Created 688 matched-pairs texts across 20 topic+form categories (7 topics × 2–3 surface forms each)
- Topic-excluded contrast sets to avoid biasing specialist identification toward syntax detection
- Four-pronged hypothesis testing: specialist specificity, representational geometry, behavioral correlation

**Key Findings:**

| Hypothesis | Question | Result |
|------------|----------|--------|
| **H1: Specialist Specificity** | Do specialists detect surface form or meaning? | ✅ SYNTAX: Mean within-topic Jaccard 0.13; specialists concentrate 96% on syntactic tokens |
| **H2: Representational Geometry** | Do representations cluster by topic or form? | ✅ TOPIC: Within-topic similarity 0.503 vs cross-topic 0.137 (p < 0.0001) |
| **H3: Behavioral Relevance** | Does specialist activation predict accuracy? | 🔬 INCONCLUSIVE: GPT-2 floor effect (2% accuracy) invalidated statistical tests |

**Central Finding: Two-Tier Representational Structure**
- **Tier 1 (Specialists):** Top 5–20 most selective features detect surface syntax (digits, operators, keywords)
- **Tier 2 (Distributed):** Thousands of weakly-active features encode semantic similarity (topic clustering)

---

### Phase 3: Cross-Model Comparison ✅ COMPLETED
**Goal:** Test whether the Phase 2 findings hold at larger model scale, and investigate the causal role of specialist features

**Colab Notebook:** [Phase 3 Colab](https://colab.research.google.com/drive/11dWRTZ9Jqf5-UQU1-Zw2osxiEF1Sn9Jh?usp=sharing)

**Corresponding Article:** *(forthcoming)* — Substack · LessWrong

**Research Question:** Does scaling from 124M to 9B parameters change how models represent meaning? Do larger models develop specialist features that detect semantic content rather than surface syntax?

**Models Compared:**

| Property | GPT-2 Small | Gemma 2 9B |
|----------|-------------|------------|
| Parameters | 124M | 9B (72x larger) |
| Layers | 12 | 42 |
| SAE vocabulary | 24,576 features | 16,384 features |
| SAE layers analyzed | 6, 8, 10, 11 | 23, 30, 37, 41 |

**Methodology:**
- Replicated all Phase 2 analyses on both models using identical matched-pairs data (688 texts)
- Extended H3 behavioral tests: 264 math tasks with calibrated difficulty, 108 non-math tasks with unambiguous answers
- Added cloze/log-probability analysis to recover signal hidden by binary scoring floor/ceiling effects
- Added multi-layer causal ablation matrix (5 specialists × 4 layers × 2 models) with random and cross-domain controls

**Key Findings:**

| Hypothesis | Result | Key Evidence |
|------------|--------|--------------|
| **H1: Specialist Specificity** | ✅ REPLICATES — Syntax, not semantics | Jaccard ~0.15–0.18 at both scales; form-dependent sparsity (ρ = 0.87) |
| **H2: Representational Geometry** | ✅ REPLICATES — Topic clustering | Within-topic > cross-topic (p < 0.0001 both models); Gemma clustering emerges only at final layer |
| **H3: Behavioral Relevance** | ⚠️ WEAKLY CAUSAL | Binary: 1/8 significant; Cloze: 6/6 significant (ρ = 0.35–0.47); Ablation: statistically significant but functionally small effects |

**Central Findings:**

1. **Scale does not solve the syntax-over-semantics problem.** Both models' specialist features detect surface syntax (Jaccard far below 1.0). The two-tier structure is scale-invariant.

2. **Larger models are dramatically more resilient to feature-level perturbation.** Gemma's signal-to-noise ratio is 140.6x vs. GPT-2's 2.6x — a 54-fold difference. Deep networks absorb and compensate for localized perturbations, with profound implications for the scalability of interpretability techniques.

3. **Selectivity does not predict causal importance.** The most selective specialist is not the most causally important (GPT-2 ρ = −0.546). The standard interpretability pipeline of identifying features by selectivity and assuming functional importance may be systematically misleading.

4. **Specialist features are weakly causal "stamps," not computational "engines."** They produce statistically significant effects when ablated (11/20 GPT-2, 15/20 Gemma conditions significant) but rarely change the predicted token (max flip rate 4.9%). The computation that drives predictions is distributed across circuits.

---

## Development Log

### [2026-02-27] - Phase 3 Completion ✅

**Cross-Model Comparison:**
- Compared GPT-2 Small (124M) and Gemma 2 9B using identical matched-pairs data
- Replicated all Phase 2 analyses; both H1 and H2 findings confirmed at 72x larger scale
- Extended H3 with calibrated difficulty math tasks, non-math tasks with unambiguous answers, cloze analysis, and multi-layer causal ablation

**Key Discoveries:**
- Two-tier structure (syntax specialists + semantic geometry) is scale-invariant
- Cloze methodology recovers signal hidden by binary scoring (6/6 significant vs. 1/8)
- Gemma 54x more resilient to single-feature ablation than GPT-2 (signal-to-noise 140.6x vs. 2.6x)
- Most selective specialist ≠ most causally important (reversed interpretability pipeline proposed)
- Ablation effects follow cascade pattern — 40.7x decay from earliest to latest layer in Gemma

**Infrastructure:**
- Migrated from local Jupyter notebooks to Google Colab (GPU required for Gemma 2 9B)
- LLM judge (Claude Sonnet API) for behavioral task scoring
- Comprehensive caching for expensive computations

### [2025-12-19] - Phase 2 Completion ✅

**Experimental Design:**
- Created matched pairs dataset (688 texts across 20 topic+form categories)
- Implemented topic-excluded contrast sets for unbiased specialist identification
- Built comprehensive visualization and statistical analysis pipeline

**Hypothesis Testing Results:**
- H1 (Individual Features): Specialists detect syntax, not semantics (mean Jaccard 0.13)
- H2 (Representational Geometry): Topic clustering confirmed (p < 0.0001)
- H3 (Behavioral): Inconclusive due to GPT-2 floor effect (2% accuracy)

**Key Insight:**
The two-tier structure — syntax-detecting specialists atop semantics-encoding distributed geometry — is a fundamental property of how SAEs decompose model representations.

### [2025-12-02] - Phase 1 Completion ✅

**Multi-SAE Comparison:**
- Systematically tested 4 SAEs across layers 6, 8, 10, 11
- Discovered layer depth strongly correlates with specialization ability
- Complete specialization (7/7 categories) achieved at layers 10-11

**Critical Bug Discovery:**
- Identified padding token masking issue that suppressed specialist detection
- Fix revealed true specialization patterns previously hidden
- Demonstrated importance of methodological rigor in interpretability research

**Key Findings:**
- Deeper layers (10-11): Complete specialization across all 7 categories
- Shallow layers (6): Only 5/7 specialists found
- Social and Conversational categories require deeper processing

### [2025-11-05] - Phase 0b Completion ✅

**Dataset Construction:**
- Created 70-text diverse dataset across 7 categories
- Each category has 10 balanced examples

**Feature Discovery Implementation:**
- Implemented 4 different feature discovery methods
- Built strongest, frequent, selective, and category-specific analyses
- Added activation strength weighting (not just binary presence)

**Specialist Search:**
- Found 1 true specialist: Feature #18522 for Math (score +6)
- Confirmed most features are general-purpose

### [2025-10-23] - Phase 0a Completion ✅

**Setup:**
- Created repository structure and Git workflow
- Set up Python virtual environment with all dependencies
- Successfully loaded GPT-2 small via TransformerLens
- Loaded pre-trained SAE (24,576 features with 32x expansion)
- Achieved 87.35% sparsity in feature extraction
- Established Neuronpedia integration for feature interpretation

---

## Contributing

This is a personal learning project, but suggestions and improvements are welcome! Feel free to open issues or submit PRs.

---

## License

MIT License - Feel free to use this code for your own learning and experimentation.

---

## Learning Resources

### Papers & Articles
- [Scaling Monosemanticity (Anthropic, 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) - Main inspiration
- [Towards Monosemanticity (Anthropic, 2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - Original SAE paper
- [Toy Models of Superposition (Anthropic, 2022)](https://transformer-circuits.pub/2022/toy_model/index.html) - Why superposition happens

### Tools & Libraries
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [SAELens GitHub](https://github.com/jbloomAus/SAELens)
- [Neuronpedia](https://neuronpedia.org/) - Explore pre-computed SAE features

---

### Related Projects
- [Mechanistic Interpretability Quickstart](https://arena3-chapter1-transformer-interp.streamlit.app/)
- [Neel Nanda's MI Resources](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

---

## Acknowledgments

- Anthropic's Interpretability Team for pioneering SAE research
- The TransformerLens and SAELens communities
- Neel Nanda for educational interpretability resources
- Joseph Bloom for pre-trained SAE releases