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

**Duration:** ~6 hours

**Key Achievements:**
- Successfully loaded GPT-2 small (768-dimensional activations)
- Loaded pre-trained SAE (24,576 features with 32x expansion)
- Extracted activations from layer 6 MLP output
- Achieved 87.35% sparsity (only 3,108/24,576 features active)
- Investigated Feature #10399 using Neuronpedia
- Saved activation cache for future use

**Key Libraries Used:**
- `transformer-lens`: Model internals access
- `sae-lens`: Pre-trained SAE loading
- `transformers`: HuggingFace base library
- `torch`: PyTorch for tensor operations
- `plotly`: Interactive visualizations

---

### Phase 2: Activation Collection & Analysis (NEXT)
**Goal:** Extract and understand activations from a target model layer

**Tasks:**
- [ ] Select target layer (likely MLP layer in middle of model)
- [ ] Extract activations from sample texts
- [ ] Visualize activation distributions and statistics
- [ ] Cache activations to disk for reuse
- [ ] Analyze activation patterns across different types of text

**Expected Duration:** 2-3 hours

**Key Concepts:**
- Hook points in transformer architecture
- MLP vs attention layer activations
- Activation caching strategies

---

### Phase 3: SAE Experimentation with Pre-trained Models
**Goal:** Use pre-trained SAEs to decompose activations into interpretable features

**Tasks:**
- [x] Load pre-trained SAE for target layer (completed in Phase 1)
- [x] Pass cached activations through SAE (completed in Phase 1)
- [x] Identify top-k active features for sample texts (completed in Phase 1)
- [ ] Find max-activating examples for interesting features
- [ ] Create visualizations showing feature activation patterns
- [ ] Build interactive exploration functions

**Expected Duration:** 2-3 hours (partially complete)

**Key Metrics:**
- Feature activation strengths
- Sparsity levels (L0 norm - how many features fire)
- Reconstruction quality

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

### Phase 5: Feature Analysis & Interpretation
**Goal:** Deep dive into what learned features represent

**Tasks:**
- [ ] Systematic feature analysis:
  - Find max-activating text examples for each feature
  - Identify feature "meanings" through dataset analysis
  - Look for polysemantic vs monosemantic features
- [ ] Build feature search/query interface
- [ ] Create feature activation heatmaps over text
- [ ] Analyze feature co-activation patterns
- [ ] Document interesting findings

**Expected Duration:** 3-5 hours

**Analysis Questions:**
- Are features interpretable to humans?
- Do features correspond to semantic concepts, syntactic patterns, or both?
- How sparse are activations in practice?
- Can we find features for specific concepts (e.g., "negation", "French", "code")?

---

### Phase 6: Interactive Demo & Documentation
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

### Phase 7: Advanced Experiments (Stretch Goals)
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

### Current Implementation (Phase 1)
```
Input Text ("The Eiffel Tower is in Paris")
    ↓
TransformerLens (GPT-2 Small)
    ↓
Extract Layer 6 MLP Activations [768-dim dense vector]
    ↓
Pre-trained SAE (from SAELens)
    ↓
Sparse Feature Activations [24,576 features, ~3,108 active = 87.35% sparse]
    ↓
Feature Analysis & Visualization
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

## Phase 1 Notebook Guide

### Overview
The Phase 1 notebook (`01_setup_and_exploration.ipynb`) demonstrates the complete pipeline for extracting activations from GPT-2 and using a pre-trained SAE to decompose them into interpretable features.

### Cell-by-Cell Breakdown

#### **Cell 1: Introduction (Markdown)**
- **Purpose**: Overview of Phase 1 objectives
- **Key Points**: Lists what you'll learn about activations, SAEs, and feature interpretation

#### **Cell 2: Import Libraries**
- **Purpose**: Import all required dependencies
- **Output**: Confirms PyTorch version (2.9.0+cpu) and device (CPU)
- **Libraries**: torch, transformer_lens, sae_lens, plotly, numpy, pandas
- **What to check**: Ensure all imports succeed without errors

#### **Cell 3: Load GPT-2 Small**
- **Purpose**: Load the target model using TransformerLens
- **Output**: 
  - Model: gpt2-small
  - Layers: 12
  - Hidden dimensions: 768
  - Attention heads: 12
- **Note**: May show `torch_dtype` deprecation warning (safe to ignore)
- **Duration**: 2-5 minutes on first run (downloads ~548MB)

#### **Cell 4: Generate Text Test**
- **Purpose**: Verify model works by generating text
- **Input**: "The capital of France is"
- **Output**: Model completes the sentence (e.g., "hosting a worldwide celebration...")
- **What to check**: Generation should be coherent and relevant

#### **Cell 5: Explore Model Architecture**
- **Purpose**: Display available hook points for activation extraction
- **Output**: 
  - 148 total parameters
  - Lists first 10 hook points
  - Explains key hooks for interpretability
- **Key Concepts**:
  - `hook_embed`: Word embeddings
  - `blocks.{X}.hook_mlp_out`: MLP layer outputs
  - `blocks.{X}.hook_resid_pre/post`: Residual stream
  - `blocks.{X}.attn.*`: Attention mechanism components

#### **Cell 6: Extract Activations**
- **Purpose**: Extract 768-dimensional activation vectors from layer 6
- **Sample texts**: 
  1. "The Eiffel Tower is in Paris"
  2. "Python is a programming language"
  3. "The cat sat on the mat"
  4. "Machine learning models learn patterns"
- **Output**: 
  - Shape for each: [768]
  - Mean: ~0.0 (centered)
  - Std: ~0.6-0.9
  - Final tensor: [4, 768]
- **What this means**: Each text becomes a 768-dimensional vector representing the model's internal understanding

#### **Cell 7: Visualize Raw Activations**
- **Purpose**: Show what dense activations look like
- **Visualization**: Heatmap of first text's 768 dimensions
- **Key Statistics**:
  - Mean: 0.0000
  - Std: 0.8513
  - Sparsity: 0.00% (no zeros)
  - Near-zero (<0.01): 1.69%
- **Interpretation**: Almost every dimension is active → DENSE → hard to interpret

#### **Cell 8: Load Pre-trained SAE**
- **Purpose**: Load a real SAE from SAELens
- **SAE Details**:
  - Source: `gpt2-small-res-jb` (Joseph Bloom's release)
  - Target: `blocks.6.hook_resid_pre`
  - Input dim: 768
  - Output dim: 24,576 (32x expansion)
- **Duration**: Should load instantly from cache (after initial download)
- **File size**: 151MB

#### **Cell 9: SAE Concept Explanation**
- **Purpose**: Illustrate the transformation concept
- **Shows**:
  - Original: 768 dims, mostly non-zero
  - SAE Encoder transforms to sparse representation
  - SAE features: 4,096 features, only ~10-20 non-zero
  - SAE Decoder reconstructs original
- **Note**: This uses simulated features for illustration

#### **Cell 9.5: Apply Real SAE** (Custom addition)
- **Purpose**: Use the loaded SAE on actual activations
- **Process**:
  1. Take first text's activation
  2. Run through SAE encoder
  3. Get sparse feature vector
- **Output**:
  - Original shape: [1, 768]
  - SAE features shape: [1, 24,576]
  - Active features: 3,108 (87.35% sparsity!)
  - Top 10 features and their activation strengths
- **What this means**: The transformation actually works! Real sparse decomposition of dense activations

#### **Cell 10: Investigate Feature #10399** (Custom addition)
- **Purpose**: Test what Feature #10399 represents
- **Method**: Run SAE on multiple test texts and see which activate the feature
- **Test texts**:
  - "The Eiffel Tower is beautiful" → 15.01
  - "Paris is the capital of France" → 12.14
  - "London has Big Ben" → 13.20
  - "I love French cuisine" → 11.65
  - "The tower was built in 1889" → 13.42
  - "Tokyo is in Japan" → 11.33
- **Finding**: Feature activates for ALL texts (not specific to Paris/Eiffel Tower)
- **Likely meaning**: General geographic/factual concept, not specific landmark

#### **Cell 11: Neuronpedia Links** (Custom addition)
- **Purpose**: Generate links to Neuronpedia for top features
- **Output**: Direct URLs to explore features:
  - Feature #10399: https://neuronpedia.org/gpt2-small/6-res-jb/10399
  - Feature #13648, #9815, #12103, #5527 (other top features)
- **Neuronpedia shows**:
  - Max-activating examples
  - Human interpretation
  - Activation patterns
- **Discovery**: Feature #10399 = "terms related to legal matters, permissions, and laws"

#### **Cell 12: Simulated Sparse Features** (Original)
- **Purpose**: Show conceptual example of sparse features
- **Output**:
  - 4,096 total features
  - 10 active (99.76% sparse)
  - Lists active features with strengths
- **Note**: This is illustrative; Cell 9.5 shows real SAE output

#### **Cell 13: Dense vs Sparse Visualization**
- **Purpose**: Visual comparison of dense and sparse representations
- **Part 1 - Bar Chart**:
  - Blue bars: Original activation (first 100 dims)
  - Orange bars: SAE features (first 100 dims)
  - Shows density difference visually
- **Part 2 - Statistical Analysis**:
  - Original: 755/768 (98.3%) significantly active
  - SAE: 3,108/24,576 (12.65%) active
  - Distribution histogram showing where active features are located
- **Key Insight**: Dense activations hard to interpret; sparse features enable interpretation

#### **Cell 14: Feature Interpretation Example**
- **Purpose**: Explain the feature interpretation methodology
- **Example Process**:
  1. Find texts that maximally activate a feature
  2. Look for patterns
  3. Hypothesize meaning
  4. Test hypothesis
- **Simulated example**: Feature representing "Paris" concepts
- **Note**: This is what researchers do to understand features

#### **Cell 15: Summary & Next Steps**
- **Purpose**: Recap what was learned
- **Key Achievements**:
  1. Loaded models with TransformerLens
  2. Extracted activations from specific layers
  3. Understood dense polysemantic activations
  4. Saw SAE decomposition into sparse features
  5. Learned feature interpretation methods
- **Key Insight**: SAEs transform uninterpretable activations → interpretable features
- **Next Steps**: Outlines Phases 2-4

#### **Cell 16: Save Progress**
- **Purpose**: Cache activations for future use
- **Saved file**: `../data/phase1_activations.pt`
- **Contents**:
  - Activations: [4, 768] tensor
  - Texts: List of 4 strings
  - Layer: 6
  - Hook name: "blocks.6.hook_mlp_out"
  - Model: "gpt2-small"
- **File size**: <1 MB
- **Usage**: Can be loaded in future notebooks to skip re-running the model

#### **Cell 17: Learning Resources**
- **Purpose**: Provide links for continued learning
- **Sections**:
  - Key papers (Anthropic's SAE research)
  - Tools (TransformerLens, SAELens, Neuronpedia)
  - Community resources (Neel Nanda, ARENA)

### Key Outputs to Understand

1. **Dense Activation (Cell 7)**
   - 768 dimensions, ~98% non-zero
   - Each dimension represents multiple concepts (polysemantic)
   - Hard to interpret what any single dimension means

2. **Sparse Features (Cell 9.5)**
   - 24,576 dimensions, only ~13% non-zero (87.35% sparse)
   - Each active feature represents one concept (monosemantic)
   - Feature #10399 = "legal/formal language" (verified via Neuronpedia)

3. **Transformation Proof (Cell 13)**
   - Visual and statistical evidence of sparse transformation
   - From 755 active dimensions → 3,108 active features
   - Higher total dimensions but much sparser (more interpretable)

### Common Questions

**Q: What's the difference between "activations" and "vectors"?**
A: Same thing! "Activation" is ML terminology; "vector" is math terminology. Both refer to the array of numbers output by a layer.

**Q: What is an SAE?**
A: Two meanings:
1. The tool/model that transforms activations (like a machine)
2. The sparse features output by that tool (like the result)

**Q: How do we know what features mean?**
A: Three methods:
1. Find texts that maximally activate each feature (most common)
2. Test feature on different texts manually
3. Use Neuronpedia where researchers have already done this analysis

**Q: Why are SAEs useful?**
A: They make interpretation possible by transforming dense, entangled representations into sparse, separated concepts. Instead of asking "what does dimension 42 mean?" (unanswerable), we can ask "what texts activate feature #10399?" (answerable: legal/formal language).

---

## Project Structure

```
interpretability-prototyping/
├── README.md                          # This file
├── notebooks/
│   └── 01_setup_and_exploration.ipynb # Phase 1 ✅ COMPLETE
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

# Open notebooks/01_setup_and_exploration.ipynb
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

**Sparse Autoencoder (SAE)**: A neural network trained to transform dense activations into sparse, interpretable features. Can refer to either the tool itself or its output.

**Dense**: Many dimensions are non-zero (e.g., 98% of values ≠ 0)

**Sparse**: Most dimensions are zero (e.g., 87% of values = 0)

**Polysemantic**: One dimension responds to multiple unrelated concepts (makes interpretation hard)

**Monosemantic**: One feature represents a single, interpretable concept (enables interpretation)

**Superposition**: Neural networks represent more concepts than they have dimensions by reusing neurons for multiple purposes

**L0 Norm**: Count of non-zero elements (how many features are active)

**L1 Sparsity**: A penalty that encourages most feature activations to be exactly zero

**Reconstruction Loss**: How well the SAE can reconstruct the original activation from sparse features

**Feature**: A dimension in the SAE's expanded representation that ideally corresponds to one interpretable concept

**Hook Point**: A location in the model where we can extract activations (e.g., `blocks.6.hook_mlp_out`)

**Residual Stream**: The main "information highway" in a transformer - accumulated information flowing through layers

**MLP (Multi-Layer Perceptron)**: The feedforward component in each transformer layer

**Attention Block**: The component that decides which tokens should attend to which other tokens

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
- Target: Layer 6 MLP output (`blocks.6.hook_mlp_out`)
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

**Next Steps:**
- Phase 2: Collect more diverse activations
- Phase 3: More systematic feature exploration
- Phase 4: Train custom SAE from scratch

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