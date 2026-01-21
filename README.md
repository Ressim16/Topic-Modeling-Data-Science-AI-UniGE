# Topic Modeling on Scientific Abstracts in Dairy Cattle Research

**Reconsidering Determinants of Health for Dairy Cattle as a Veterinary Public Health Challenge**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](acknowledgments)
- [References](references)

---

## Overview

This project applies **advanced topic modeling techniques** to analyze scientific literature on dairy cattle health from 2000 to 2025. Using data from PubMed and Scopus, we employ **BERTopic** with domain-specific embeddings (BioBERT) to uncover hidden thematic patterns in veterinary public health research.

### Key Objectives

- Extract and analyze scientific abstracts from major biomedical databases
- Apply state-of-the-art NLP and topic modeling techniques
- Identify emerging research trends in dairy cattle health
- Provide insights into the evolution of veterinary public health research

### Technologies Used

- **Topic Modeling**: BERTopic with HDBSCAN clustering
- **Embeddings**: BioBERT for domain-specific semantic representation
- **LLM Integration**: Local language models for topic refinement
- **Dimensionality Reduction**: UMAP
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly

---

## Project Structure

```
dairy-cattle-research-topic-modeling/
│
├── data/                                   # Data directory
│   ├── cleaned_combined_dataset.csv        # Main cleaned dataset
│   ├── potential_duplicates.csv            # Identified potential duplicates
│   ├── pubmed_5000.csv                     # PubMed raw data sample
│   └── scopus_5000.csv                     # Scopus raw data sample
│
├── output/                                 # Output directory
│   ├── vectors/                            # Embedding vectors
│   ├── figures/                            # Generated visualizations
│   ├── llm/                                # LLM-generated topic labels
│   └── modeling/                           # Saved models and results
│
├── Rapport_Unige_Data_Science.ipynb        # Main analysis notebook
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
└── .gitignore                              # Git ignore file
```

---

## Features

### Data Collection & Processing
- Automated data extraction from PubMed and Scopus APIs
- Robust preprocessing pipeline with duplicate detection
- Text standardization for NLP processing

### Exploratory Data Analysis
- Temporal trend analysis (2000-2025)
- Journal distribution and publication patterns
- Author contribution analysis
- Network visualization

### Advanced Topic Modeling
- **BERTopic** implementation with customizable parameters
- **BioBERT embeddings** for biomedical text understanding
- **UMAP** for dimensionality reduction
- **HDBSCAN** for density-based clustering
- **LLM-powered topic refinement** with structured JSON output

### Visualization
- Interactive topic maps
- Temporal evolution of research themes
- Publication distribution across journals
- Cluster quality metrics

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- (Optional) GPU for faster embedding generation

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Ressim16/Topic-Modeling-Data-Science-AI-UniGE
cd Topic-Modeling-Data-Science-AI-UniGE
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models** (if not automatically downloaded)
```bash
# BioBERT embeddings will be downloaded on first run
# LLM models will be cached locally
```

---

## Usage

### Running the Analysis

1. **Open Jupyter Notebook**
```bash
jupyter notebook Rapport_Unige_Data_Science.ipynb
```

2. **Execute cells sequentially**
   - Follow the notebook from data collection through analysis
   - Modify parameters as needed for your use case

### Key Parameters to Adjust

```python
# Topic modeling parameters
n_topics = 20  # Number of topics to extract
min_cluster_size = 15  # Minimum cluster size for HDBSCAN
n_neighbors = 15  # UMAP neighbors parameter

# Embedding model
embedding_model = "pritamdeka/S-BioBert-snli-multinli-stsb"  # BioBERT

# LLM for topic refinement
llm_model = "meta-llama/Llama-3.2-1B-Instruct"  # Adjust based on compute resources
```

---

## Data

### Data Sources

**PubMed** (U.S. National Library of Medicine)
- Access via PubMed E-utilities API
- Search terms: Dairy cattle health-related keywords
- Time range: 2000-2025
- Sample size: 28,813

**Scopus** (Elsevier)
- Access via Scopus API
- Search terms: Dairy cattle health-related keywords
- Time range: 2000-2025
- Sample size: 45,190 abstracts

### Dataset Characteristics

- **Combined dataset**: 74,003 scientific abstracts (restricted to 10,000 in this project)
- **Fields**: Title, Abstract, Authors, Journal, Year, DOI
- **Language**: Primarily English
- **Quality**: Deduplicated and preprocessed

### Data Availability

Due to licensing restrictions, raw data from PubMed and Scopus is not included in this repository. Users must:
1. Obtain API access from respective databases
2. Obtain targeted data from specific domain research and related features
3. Ensure compliance with database terms of service

---

## Methodology

### Pipeline Overview

```
Data Collection → Preprocessing → EDA → Topic Modeling → Visualization → Interpretation
```

### 1. Data Collection
- API-based extraction from PubMed and Scopus
- Standardized data formatting
- Metadata preservation

### 2. Preprocessing
- Text cleaning and standardization
- Duplicate detection and removal
- Language filtering

### 3. Exploratory Data Analysis
- Temporal trend analysis
- Publication pattern identification
- Journal distribution analysis
- Author network exploration

### 4. Topic Modeling (BERTopic)
- **Embedding**: BioBERT for domain-specific representation
- **Dimensionality Reduction**: UMAP for efficient clustering
- **Clustering**: HDBSCAN for density-based topic discovery
- **Representation**: c-TF-IDF for keyword extraction
- **Refinement**: LLM-based topic labeling with structured output

---

## Results

### Key Findings

1. **Publication Growth**: Research in dairy cattle health increased from 71 publications in 2000 to 1,011 in 2024

2. **Journal Concentration**: Top 10 journals account for ~20% of all publications

3. **Topic Diversity**: 20 distinct research themes identified, covering areas such as:
   - Reproductive health
   - Infectious diseases
   - Nutrition and metabolism
   - Welfare and behavior
   - Production optimization
   - ...

### Visualizations

See the `output/figures/` directory for:
- Author publication trend
- Temporal evolution charts
- Journal publication patterns

---

## Limitations

### Data-Related Limitations

- **Language Bias**: Predominantly English publications; potential underrepresentation of non-English research
- **Abstract-Only**: Analysis limited to abstracts, missing full-text nuances
- **Single Author Per Paper**: Scopus constraint prevents comprehensive author network analysis

### Methodological Limitations

- **Hyperparameter Sensitivity**: Numerous parameters in BERTopic pipeline difficult to optimize exhaustively
- **Computational Constraints**: Limited to smaller LLM models, potentially affecting topic refinement quality, notably with hallucinations
- **Outlier Assignment**: Conservative clustering may miss small but significant research niches
- **Reduced Granularity**: Limited to 20 topics; finer-grained analysis may reveal additional insights
- **Sample Size**: Representative sample of 5 papers per cluster for LLM refinement may not capture full cluster diversity

### Validation Challenges

- No ground truth labels for quantitative validation
- Topic interpretability relies on subjective assessment
- Domain expert validation recommended but not included

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Improved hyperparameter optimization
- Additional data sources
- Enhanced visualization techniques
- Validation frameworks
- Performance optimizations
- Documentation improvements

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Project Author**: Reda Zahri
- Email: reda.zahri@hotmail.com
- GitHub: [@Ressim16](https://github.com/Ressim16)
- LinkedIn: [Reda Zahri](https://www.linkedin.com/in/reda-zahri/)

**Institution**: University of Geneva
- Course: Data Science, Machine Learning & AI
- Department: Centre Universitaire d'Informatique (CUI)

---

## Acknowledgments

- University of Geneva for academic support
- PubMed and Scopus for data access
- BERTopic and Hugging Face communities
- BioBERT authors for domain-specific embeddings

---

## References

### Key Libraries
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [BioBERT](pritamdeka/S-BioBert-snli-multinli-stsb)
- [UMAP](https://umap-learn.readthedocs.io/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [LLM](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

### Relevant Papers
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
- Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model.

---

**Last Updated**: January 2026

---

**AI-Use in this project**: All the texts and codes were refined with the help of [Claude Chatbot](https://claude.ai/)
