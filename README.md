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

## Current Status

✅ **Phase 1: Initial Setup & Exploration** - COMPLETED (2025-10-23)
✅ **Phase 2: Feature Exploration & Analysis** - COMPLETED (2025-11-05)

---

## Roadmap

### Phase 1: Environment Setup & Exploration ✅ COMPLETED
**Goal:** Get familiar with existing interpretability tools using off-the-shelf libraries

**Tasks:**
- [x] Initialize Git repository and create project structure
- [x] Set up Python environment with required dependencies
- [x] Create initial Jupyter notebook for experimentation
- [x] Load a small pre-trained model (GPT-2 small) using TransformerLens
- [x] Explore pre-trained SAEs using SAELens library
- [x] Visualize basic activation patterns
- [x] Use Neuronpedia to understand what interpretable features look like
- [x] Successfully load and use real pre-trained SAE
- [x] Extract and analyze activations from GPT-2
- [x] Compare dense vs sparse representations
- [x] Investigate specific features and their meanings
- [x] Extract activations from sample texts
- [x] Cache activations to disk for reuse

**Duration:** ~6 hours

**Key Achievements:**
- Successfully loaded GPT-2 small (768-dimensional activations)
- Loaded pre-trained SAE (24,576 features with 32x expansion)
- Extracted activations from Layer 6 residual stream (`blocks.6.hook_resid_pre`)
- Achieved 87.35% sparsity (only 3,108/24,576 features active)
- Investigated Feature #10399 using Neuronpedia
- Saved activation cache for future use
- Extracted and cached activations from 4 initial sample texts

**Key Libraries Used:**
- `transformer-lens`: Model internals access
- `sae-lens`: Pre-trained SAE loading
- `transformers`: HuggingFace base library
- `torch`: PyTorch for tensor operations
- `plotly`: Interactive visualizations

---

### Phase 2: Feature Exploration & Analysis ✅ COMPLETED
**Goal:** Systematically explore and analyze SAE features to understand what patterns they capture

**Tasks:**
- [x] Build diverse test dataset (70 texts across 7 categories)
- [x] Extract features from diverse text types (Python, URLs, Math, Non-English, Social/Emoji, Formal, Conversational)
- [x] Implement multi-criteria feature discovery:
  - Strongest features (highest activation values)
  - Most frequent features (activate across most texts)
  - Most selective features (high activation on few texts)
  - Category-specific analysis (search for specialists)
- [x] Create feature activation heatmaps
- [x] Build interactive feature explorer function
- [x] Analyze feature co-activation patterns
- [x] Find features for specific concepts (with activation strength weighting)
- [x] Integrate Neuronpedia links throughout analysis
- [x] Implement SAE selection mechanism for comparing different decompositions

**Duration:** ~8 hours

**Key Achievements:**

**Dataset Construction:**
- Created 70-text diverse dataset spanning 7 categories
- 10 texts per category ensuring balanced representation
- Categories: Python code, URLs/web content, mathematical notation, non-English languages, social media/emoji, formal writing, conversational English

**Feature Discovery Methods:**
- **Strongest Feature**: Feature #10399 (max activation 16.85) - activates on legal text, greetings, and formal language
- **Most Frequent Feature**: Feature #174 (active in 69/70 texts) - extremely general background feature
- **Most Selective Feature**: Feature #174 (strong in only 1/70 texts) - rare but impactful activation
- **Category Specialists**: Found 1 true specialist - Feature #18522 for Math (strong activations in 7/10 Math texts and only 1/60 non-Math texts)

**Key Insights:**
- Most features in the pre-trained SAE (6-res-jb, layer 6 residual stream) are general-purpose
- True category specialists are rare - only 1 found across 7 categories
- Feature #18522 demonstrated genuine specialization for mathematical notation
- General features like #18, #31, #45 activate consistently across all text types
- Composite scoring (frequency × mean activation) provides better ranking than binary presence/absence

**Visualizations:**
- Feature activation heatmaps showing patterns across 70 diverse texts
- Category-wise activation analysis with mean activation by category
- Top 10 frequent features visualization
- Interactive feature explorer for detailed investigation

**Interpretability Findings:**
- SAEs don't always learn clean, monosemantic specialists for every domain
- Most features represent general linguistic patterns (punctuation, structure, syntax)
- Domain-specific patterns (math, code) are captured by combinations of features rather than single specialists
- Looking at activation strength (not just presence) reveals more nuanced feature behavior

**Technical Improvements:**
- Implemented activation strength weighting in concept finding
- Added composite scoring (frequency × strength) for better feature ranking
- Built framework for comparing multiple SAEs (preparation for future work)
- Structured code for easy SAE selection and comparison

**Expected Next Steps:**
- Test additional pre-trained SAEs to find more specialists
- Compare how different SAEs decompose the same activations
- Investigate why some SAEs learn specialists while others learn general features

---

### Phase 3: SAE Comparison & Advanced Analysis (PLANNED)
**Goal:** Compare multiple pre-trained SAEs and understand how training affects feature specialization

**Tasks:**
- [ ] Load and test multiple SAEs (different layers, different training runs)
- [ ] Run Phase 2 analysis pipeline on each SAE
- [ ] Compare specialist features across SAEs
- [ ] Analyze which SAE architectures/training produce better specialists
- [ ] Document differences in feature interpretability

**Expected Duration:** 3-4 hours

**Key Questions:**
- Do deeper layers (8, 10) learn more specialized features than layer 6?
- Do MLP-output SAEs differ from residual-stream SAEs?
- Can we find code specialists, emoji specialists, or other domain experts?

---

### Phase 4: Training a Custom SAE (Homebrew Version)
**Goal:** Understand SAE internals by implementing and training one from scratch

**Tasks:**
- [ ] Implement SAE architecture:
  - Encoder: Linear layer + ReLU activation
  - Decoder: Linear layer (tied weights optional)
- [ ] Implement training loop with:
  - Reconstruction loss (MSE between input and reconstructed activations)
  - L1 sparsity penalty
  - Optional auxiliary losses
- [ ] Train on collected activations
- [ ] Monitor training metrics (loss, sparsity, reconstruction quality)
- [ ] Compare custom SAE performance to pre-trained versions
- [ ] Tune hyperparameters (L1 coefficient, learning rate, SAE width)

**Expected Duration:** 4-6 hours

**Key Hyperparameters:**
- `d_model`: Input dimension (e.g., 768 for GPT-2 small)
- `d_sae`: SAE feature dimension (typically 4-8x d_model)
- `l1_coeff`: Sparsity penalty strength (typically 1e-3 to 1e-2)
- `learning_rate`: Training learning rate (typically 1e-4)

---

### Phase 5: Interactive Demo & Documentation
**Goal:** Package work into reusable, well-documented format

**Tasks:**
- [ ] Refactor notebook code into modular Python files:
  - `models.py`: Model loading and activation extraction
  - `sae.py`: SAE architecture and training
  - `analysis.py`: Feature analysis utilities
  - `visualization.py`: Plotting and display functions
  - `data.py`: Dataset handling and caching
- [ ] Create clean demo notebook showcasing key capabilities
- [ ] Write comprehensive documentation
- [ ] Add example outputs and visualizations to README
- [ ] (Optional) Build simple Streamlit/Gradio interface

**Expected Duration:** 3-4 hours

---

### Phase 6: Advanced Experiments (Stretch Goals)
**Goal:** Explore advanced interpretability techniques

**Possible Extensions:**
- [ ] Steering: Modify model behavior by amplifying/suppressing features
- [ ] Feature attribution: Which features matter most for specific predictions?
- [ ] Cross-layer analysis: How features evolve across layers
- [ ] Automated interpretability: Use an LLM to describe features
- [ ] Causal interventions: Ablate features and measure impact
- [ ] Compare interpretability across different model sizes/architectures

---

## Technical Architecture

### Current Implementation (Phase 1-2)
```
Input Text (70 diverse examples across 7 categories)
    ↓
TransformerLens (GPT-2 Small)
    ↓
Extract Layer 6 Residual Stream Activations [768-dim dense vector]
    ↓
Pre-trained SAE (from SAELens) - selectable from multiple options
    ↓
Sparse Feature Activations [24,576 features, ~87% sparse]
    ↓
Feature Analysis:
    - Strongest features (highest activation)
    - Most frequent features (appear in most texts)
    - Most selective features (high activation, rare occurrence)
    - Category specialists (strong in one category, weak elsewhere)
    ↓
Visualizations & Interactive Exploration
```

### Future Approach (Phase 4+)
```
Input Text
    ↓
Model → Activations → Cache Dataset
    ↓
Custom SAE Training
    ↓
Trained SAE → Feature Analysis → Insights
```

---

## Phase 2 Notebook Guide

### Overview
The Phase 2 notebook (`phase_2_feature_exploration.ipynb`) demonstrates systematic feature exploration and analysis across diverse text types to understand what patterns the SAE has learned.

### Cell-by-Cell Breakdown

#### **Cell 1: Introduction (Markdown)**
- **Purpose**: Overview of Phase 2 objectives
- **Key Points**: Systematic exploration, max-activating examples, visualization, interactive tools, co-activation patterns

#### **Cell 2: Import Libraries**
- **Purpose**: Import all required dependencies
- **Libraries**: torch, numpy, pandas, plotly, transformer-lens, sae-lens
- **Setup**: Random seeds for reproducibility
- **Output**: Confirmation of successful imports and PyTorch version

#### **Cell 3: Load LLM & SAE (with Multiple SAE Options)**
- **Purpose**: Load GPT-2 model and select from multiple pre-trained SAEs
- **Key Features**:
  - Dictionary of available SAEs with metadata
  - Easy SAE selection via variable
  - Automatic hook point configuration
  - Error handling for missing SAEs
- **Available SAEs**:
  - `6-res-jb`: Layer 6 Residual Stream (default)
  - `8-res-jb`: Layer 8 Residual Stream
  - `10-res-jb`: Layer 10 Residual Stream
  - `6-mlp-out`: Layer 6 MLP Output
- **Output**: Model loaded, SAE selection confirmed, dimensions displayed

#### **Cell 4: Extract Features from Texts**
- **Purpose**: Extract and display features from initial sample texts
- **Process**:
  - Run 4 sample texts through model
  - Extract layer 6 activations
  - Pass through SAE to get sparse features
  - Display top 5 features for each text
- **Output**: Feature activations for each text with indices and values
- **Key Insight**: Same features (like #5527, #9815, #10399) appear across different texts

#### **Cell 5: Find Most Frequently Active Features**
- **Purpose**: Identify features that activate consistently across all texts
- **Method**: Count how many texts activate each feature (activation > 0)
- **Parameters**: Configurable `most_freq_features` variable (default 20)
- **Output**: 
  - Ranked list of most frequent features
  - Activation counts and percentages
  - Interpretation of universal vs specialized features
- **Key Finding**: Top features often activate in 100% of texts (general-purpose)

#### **Cell 6: Load Larger Diverse Dataset**
- **Purpose**: Create comprehensive test dataset spanning multiple domains
- **Dataset Size**: 70 texts across 7 categories (10 per category)
- **Categories**:
  - Python code (functions, imports, loops, classes)
  - URLs/Web content (links, HTML, HTTP requests)
  - Mathematical notation (equations, integrals, limits, formulas)
  - Non-English languages (French, Chinese, Spanish, German, Russian, Japanese, Arabic, Korean, Italian, Portuguese)
  - Social media/emoji (casual slang, emojis, internet speak)
  - Formal/academic writing (research language, legal text)
  - Conversational English (everyday phrases)
- **Process**: Extract features for all 70 texts
- **Output**: Feature tensor [70, 24576] with category labels

#### **Cell 7: Discover and Analyze Most Interesting Features**
- **Purpose**: Find and analyze features using multiple criteria
- **Approach 1 - Strongest Feature**:
  - Finds feature with highest single activation value
  - Example: Feature #10399 (16.85)
- **Approach 2 - Most Frequent Feature**:
  - Finds feature active in most texts
  - Example: Feature #174 (active in 69/70 texts)
- **Approach 3 - Most Selective Feature**:
  - Finds features with high activation (>5.0) in fewest texts
  - Uses activation threshold to identify rare specialists
- **Approach 4 - Category-Specific Analysis**:
  - Searches for specialists in each of 7 categories
  - Calculates specialist score: (strong activations inside category) - (strong activations outside)
  - Example finding: Feature #18522 for Math (score: +6)
  - Identifies true specialists vs general features
- **Output**: 
  - Detailed analysis of top 3 features with statistics
  - Category breakdown with Neuronpedia links
  - Specialist scores and interpretation
  - Summary of findings (e.g., "1/7 categories have specialist features")
- **Key Insight**: Most features are general-purpose; true specialists are rare

#### **Cell 8: Feature Activation Heatmap**
- **Purpose**: Visualize activation patterns across diverse texts
- **Visualizations**:
  1. **Heatmap**: Shows feature activations across all 70 texts
     - Rows: Top 10 frequent features
     - Columns: All 70 texts with category labels
     - Color intensity: Activation strength
  2. **Bar Chart**: Mean activation by category for each feature
     - Groups bars by category
     - Compares feature behavior across domains
- **Output**: 
  - Interactive Plotly visualizations
  - Interpretation guide
  - Dynamic messaging based on whether specialists were found
- **Key Insight**: Visual confirmation of general vs specialized features

#### **Cell 9: Build Interactive Feature Explorer Function**
- **Purpose**: Create reusable tool for exploring individual features
- **Function**: `explore_feature(feature_idx, num_examples=10)`
- **Features**:
  - Displays Neuronpedia link
  - Shows top N activating texts
  - Returns sorted list of (text, activation) pairs
- **Example**: Explores Feature #10399 (strongest from Cell 7)
- **Output**: 
  - Ranked list of texts that activate the feature
  - Activation values for each text
  - Full dataset output for further analysis
- **Usage**: Can be called on any feature index for investigation

#### **Cell 10: Feature Co-Activation Analysis**
- **Purpose**: Identify which features consistently activate together
- **Method**: For each category, find features active in most texts within that category
- **Output**: 
  - Top 5 most common features per category with Neuronpedia links
  - Activation frequencies within each category
- **Key Finding**: Same general features (e.g., #18, #31, #36, #45) appear across all categories
- **Interpretation**: Confirms lack of category-specific specialists in this SAE

#### **Cell 11: Find Features for a Specific Concept (with Activation Strength)**
- **Purpose**: Find features that characterize a specific concept (e.g., "Code/Programming")
- **Method**: 
  - Analyzes multiple example texts of the concept
  - Counts strong activations (>5.0) per feature
  - Calculates mean activation across all examples
  - Ranks by composite score (frequency% × mean activation)
- **Parameters**:
  - `concept_texts`: List of example texts
  - `top_k`: Number of features to return (default 5)
  - `activation_threshold`: Minimum for "strong" activation (default 5.0)
- **Example**: "Code/Programming" concept with 5 Python snippets
- **Output**:
  - Top features ranked by composite score
  - Strong frequency (% of texts with activation >5.0)
  - Mean activation across all texts
  - Neuronpedia links
- **Key Improvement**: Considers both frequency AND strength (not just binary presence)
- **Results**: 
  - Feature #18: 100% frequency, 9.92 mean, 992 composite score
  - Feature #31: 100% frequency, 9.85 mean, 985 composite score
  - Feature #36: 100% frequency, 4.97 mean, 497 composite score (notably weaker)

### Key Findings from Phase 2

**Specialist Features:**
- Only 1 true specialist found: Feature #18522 for mathematical notation
- Specialist score of +6 (activates strongly in Math but rarely elsewhere)
- Most other features are general-purpose across all categories

**General Features:**
- Features #18, #31, #36, #45 activate across all text types
- These represent fundamental linguistic patterns rather than domain-specific concepts
- Activation strength varies but presence is consistent

**Activation Patterns:**
- Composite scoring (frequency × strength) reveals meaningful differences
- Binary presence/absence masks important nuances
- Some features activate everywhere but with different intensities

**SAE Behavior:**
- The 6-res-jb SAE learned primarily general features
- True specialists are rare in this particular decomposition
- Different SAEs (layers, training) may learn different feature types

**Interpretability Challenges:**
- Finding monosemantic specialists is harder than expected
- Most features capture combinations of patterns rather than single concepts
- Manual analysis with diverse datasets is essential for understanding features

---

## Project Structure

```
interpretability-prototyping/
├── README.md                          # This file
├── notebooks/
│   ├── phase_1.ipynb                  # Phase 1 ✅ COMPLETE
│   └── phase_2_feature_exploration.ipynb # Phase 2 ✅ COMPLETE
├── src/                               # Future: Refactored code
├── data/                              
│   ├── phase1_activations.pt          # Cached activations from Phase 1
│   └── .gitignore
├── outputs/                           
│   └── .gitignore
├── requirements.txt                   # Python dependencies
└── .gitignore
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- 8GB+ RAM
- Internet connection for downloads

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformer-lens sae-lens
pip install transformers datasets einops
pip install plotly jupyter ipywidgets
pip install numpy pandas matplotlib
```

### Quick Start

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/phase_1.ipynb
# 2. notebooks/phase_2_feature_exploration.ipynb
# Run cells sequentially (Shift+Enter)
```

### Troubleshooting

**Model download stalls:**
- Use terminal to download: `huggingface-cli download openai-community/gpt2`
- Copy files to proper cache location

**SAE loading hangs:**
- Use terminal download then `SAE.load_from_disk(path)`
- Ensure files are in `~/.cache/sae_lens/`

**Import errors:**
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

---

## Key Concepts & Terminology

**Activation / Vector**: The output of a specific layer - an array of numbers representing the model's internal state. These terms are interchangeable.

**Sparse Autoencoder (SAE)**: A neural network trained to transform dense activations into sparse, interpretable features. Can refer to either the tool itself or its output. The SAE is pre-trained (not trained by you) and acts like a "key" that knows how to unfold/disaggregate the compressed activation vector.

**Dense**: Many dimensions are non-zero (e.g., 98% of values ≠ 0)

**Sparse**: Most dimensions are zero (e.g., 87% of values = 0)

**Polysemantic**: One dimension responds to multiple unrelated concepts (makes interpretation hard)

**Monosemantic**: One feature represents a single, interpretable concept (enables interpretation)

**Specialist Feature**: A feature that activates strongly in one domain/category but rarely in others. True specialists are rare and valuable for interpretability.

**General Feature**: A feature that activates across many text types. Represents fundamental linguistic patterns rather than domain-specific concepts.

**Superposition**: Neural networks represent more concepts than they have dimensions by reusing neurons for multiple purposes

**L0 Norm**: Count of non-zero elements (how many features are active)

**L1 Sparsity**: A penalty that encourages most feature activations to be exactly zero

**Reconstruction Loss**: How well the SAE can reconstruct the original activation from sparse features

**Feature**: A dimension in the SAE's expanded representation that ideally corresponds to one interpretable concept

**Hook Point**: A location in the model where we can extract activations (e.g., `blocks.6.hook_resid_pre`)

**Residual Stream**: The main "information highway" in a transformer - accumulated information flowing through layers

**MLP (Multi-Layer Perceptron)**: The feedforward component in each transformer layer

**Attention Block**: The component that decides which tokens should attend to which other tokens

**Composite Score**: In feature ranking, the product of activation frequency and mean activation strength, balancing consistency with intensity

**Specialist Score**: (strong activations inside category) - (strong activations outside category). Positive scores indicate true specialists.

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

### Related Projects
- [Mechanistic Interpretability Quickstart](https://arena3-chapter1-transformer-interp.streamlit.app/)
- [Neel Nanda's MI Resources](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

---

## Progress Log

### [2025-11-05] - Phase 2 Completion ✅
**Duration**: ~8 hours

**Dataset Construction:**
- Created 70-text diverse dataset across 7 categories
- Each category has 10 balanced examples
- Categories: Python, URLs/Web, Math, Non-English, Social/Emoji, Formal, Conversational

**Feature Discovery Implementation:**
- Implemented 4 different feature discovery methods
- Built strongest, frequent, selective, and category-specific analyses
- Added activation strength weighting (not just binary presence)
- Implemented composite scoring (frequency × strength)

**Specialist Search:**
- Systematically searched for specialists in each category
- Found 1 true specialist: Feature #18522 for Math (score +6)
- Confirmed most features are general-purpose (activate across categories)

**Visualizations:**
- Created interactive heatmaps showing 70 texts × features
- Built category-wise mean activation bar charts
- Integrated Neuronpedia links throughout

**Interactive Tools:**
- Built `explore_feature()` function for investigating individual features
- Implemented co-activation analysis showing which features fire together
- Created `find_features_for_concept()` with strength weighting

**Key Technical Improvements:**
- Moved from binary activation (on/off) to strength-based ranking
- Implemented specialist scoring: (inside - outside) activations
- Added SAE selection framework for future comparisons
- Structured code for systematic analysis across multiple SAEs

**Interpretability Insights:**
- SAEs don't always learn perfect monosemantic specialists
- Most features are general (punctuation, structure, syntax)
- Domain-specific patterns emerge from feature combinations
- Activation strength matters as much as frequency

**Challenges & Solutions:**
- Finding specialists was harder than expected → systematized search across categories
- Binary presence insufficient → added strength weighting
- Needed better ranking → implemented composite scores
- Wanted SAE comparison → built selection framework

### [2025-10-23] - Phase 1 Completion ✅
**Duration**: ~6 hours

**Setup:**
- Created repository structure
- Configured Git workflow (main + feature branches)
- Set up Python virtual environment
- Installed all dependencies (transformer-lens, sae-lens, torch, plotly, etc.)
- Resolved Chrome browser configuration in WSL

**Model Loading:**
- Successfully loaded GPT-2 small via TransformerLens
- Overcame download progress bar issues in Jupyter
- Verified model functionality with text generation

**Activation Extraction:**
- Extracted activations from 4 sample texts
- Target: Layer 6 residual stream (`blocks.6.hook_resid_pre`)
- Shape: [4, 768] - four 768-dimensional vectors
- Visualized dense activation patterns

**SAE Loading & Analysis:**
- Downloaded pre-trained SAE (151MB) from SAELens
- Loaded SAE: 768 input → 24,576 features (32x expansion)
- Successfully transformed activations → achieved 87.35% sparsity
- Only 3,108 out of 24,576 features active per input

**Feature Investigation:**
- Analyzed Feature #10399 with multiple test texts
- Used Neuronpedia to discover true meaning: "legal/formal language"
- Generated links for top-10 active features
- Demonstrated feature interpretation methodology

**Visualization:**
- Created dense vs sparse comparison charts
- Generated statistical analysis of activation patterns
- Showed distribution of active features across 24,576 dimensions

**Data Persistence:**
- Saved activation cache to `data/phase1_activations.pt`
- Documented file contents and usage for future phases

**Key Learnings:**
- Dense activations (98% non-zero) are uninterpretable
- SAEs transform to sparse features (87% zero) enabling interpretation
- Each SAE feature represents a single concept (monosemantic)
- Neuronpedia is essential for understanding pre-analyzed features
- Feature interpretation requires testing on many examples

**Challenges Overcome:**
- Model download stalls → solved with terminal downloads
- Browser configuration → set Chrome as default in WSL
- SAE loading hangs → used `load_from_disk` with cached files
- Progress bar errors → cosmetic issue, downloads worked

---

## Contributing

This is a personal learning project, but suggestions and improvements are welcome! Feel free to open issues or submit PRs.

---

## License

MIT License - Feel free to use this code for your own learning and experimentation.

---

## Acknowledgments

- Anthropic's Interpretability Team for pioneering SAE research
- The TransformerLens and SAELens communities
- Neel Nanda for educational interpretability resources
- Joseph Bloom for pre-trained SAE releases