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
4. Create a foundation for more advanced interpretability experiments

---

## Getting Started

### Prerequisites
- Python 3.9+
- 8GB+ RAM
- Internet connection for initial model/SAE downloads

### 1. Clone the Repository
```bash
git clone <repository-url>
cd interpretability-prototyping
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/Mac/WSL)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
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

### 4. Download Pre-trained SAEs (Optional but Recommended)
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

### 5. Launch Jupyter and Run Notebooks

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

### 6. Run Notebooks in Order

The notebooks are designed to be run sequentially:

| Notebook | Description |
|----------|-------------|
| `notebooks/phase_0a.ipynb` | Initial setup, model loading, basic SAE exploration |
| `notebooks/phase_0b_feature_exploration.ipynb` | Feature analysis, heatmaps, specialist search |
| `notebooks/phase_1_SAE_comparison.ipynb` | Multi-SAE comparison across layers |
| `notebooks/phase_2_semantics_vs_ideas.ipynb` | Feature activation: syntax vs. meaning |

### Troubleshooting

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

## Current Status

✅ **Phase 0a: Initial Setup & Exploration** - COMPLETED (2025-10-23)
✅ **Phase 0b: Feature Exploration & Analysis** - COMPLETED (2025-11-05)
✅ **Phase 1: SAE Comparison & Advanced Analysis** - COMPLETED (2025-12-02)
✅ **Phase 2: Semantics vs. Syntax Analysis** - COMPLETED (2025-12-19)


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

**Corresponding Article:** "A Peek Inside the Black Box: Part 1"

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

**Corresponding Article:** "A Peek Inside the Black Box: Part 2"

**Research Question:** Do SAE specialist features respond to the underlying *concept* (semantic meaning) or merely the *surface form* (syntactic patterns like operators and symbols)?

**Methodology:**
- Created matched pairs dataset: 18 mathematical expressions in three surface forms
  - Symbolic: `x^2 + 2x + 1`
  - Verbal: `x squared plus two x plus one`
  - Prose: `the square of x added to twice x and one`
- Tested four hypotheses examining different aspects of the syntax-vs-semantics question

**Key Findings:**

| Hypothesis | Question | Result |
|------------|----------|--------|
| **H1: Specialist Activation** | Do specialists discriminate by form? | ✅ SUPPORTED: 6-11× higher activation for symbolic vs. verbal forms |
| **H2: Feature Population** | Do forms activate different feature groups? | ✅ SUPPORTED: Verbal-Prose overlap (63%) >> Symbolic-Verbal overlap (40-48%) |
| **H3: Representational Geometry** | Do representations cluster by form or concept? | ✅ SUPPORTED: Within-form similarity (0.53) > Within-concept similarity (0.40), p < 0.001 |
| **H4: Behavioral Correlation** | Does activation predict accuracy? | ❌ INCONCLUSIVE: GPT-2 Small achieved 0% accuracy on all forms (task too difficult) |

**Central Finding:**
Specialist features in GPT-2 Small detect **syntactic patterns** (arithmetic operators like +, =, ^) rather than **semantic concepts**. The model's internal representations cluster by surface form, not by underlying meaning. This challenges optimistic interpretations that SAE features represent genuine conceptual understanding.

**Implications:**
- SAE "specialists" learn efficient low-level patterns that correlate with human-defined categories, not the categories themselves
- Conceptual understanding, if present, may reside elsewhere in the architecture (attention patterns, distributed representations)
- The threshold for genuine conceptual feature emergence may require larger models than GPT-2 Small (124M parameters)

---

## Development Log

### [2025-12-19] - Phase 2 Completion ✅

**Experimental Design:**
- Created matched pairs dataset (18 math expressions × 3 forms = 54 texts)
- Implemented four-pronged hypothesis testing framework
- Built comprehensive visualization and statistical analysis pipeline

**Hypothesis Testing Results:**
- H1 (Individual Features): Symbolic forms show 6-11× higher specialist activation
- H2 (Feature Populations): Natural language forms share 63% overlap; symbolic shares only 40-48%
- H3 (Representational Geometry): Clustering by form confirmed (p < 0.001)
- H4 (Behavioral): Inconclusive due to task difficulty for GPT-2 Small

**Key Insight:**
The findings align with what Anthropic found in toy models—specialist features detect syntax, not semantics—but this phase demonstrates the pattern persists even in a 124M parameter production model.

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