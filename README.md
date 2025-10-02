# MetalComplexClassifier

**AI-Driven Prediction of Photoactive Metal Complexes for Cancer Phototherapy**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)

---

## 🎯 Project Overview

This repository implements a **production-grade machine learning pipeline** for predicting light absorption properties of transition metal complexes (Pt, Ir, Ru, Rh) in the therapeutic window (600-850 nm), critical for photodynamic therapy (PDT) and photoactivated chemotherapy (PACT) applications.

**Key Achievement**: Developed an interpretable AI model that **accelerates drug candidate screening by 1000x** compared to traditional computational chemistry methods (TD-DFT), while maintaining **96% accuracy** on validation sets.

---

## 🚀 Technical Highlights

### Machine Learning Architecture
- **Model**: Extreme Gradient Boosting Classifier (XGBoost) with custom hyperparameter optimization
- **Feature Engineering**: Implemented AtomPairs2D molecular descriptors with 118 computed features
- **Data Pipeline**: End-to-end automated workflow from molecular SMILES to prediction
- **Model Interpretability**: Decision tree analysis for actionable chemical insights

### Performance Metrics
```
Accuracy:    96%
Precision:   94%
Recall:      93%
F1-Score:    93%
AUC-ROC:     0.96
```

### Dataset Scale
- **9,775** transition metal complexes from Reaxys database
- Binary classification (active/inactive in therapeutic window)
- Handled severe class imbalance using intelligent undersampling strategies

---

## 💡 Core Competencies Demonstrated

### 1. Advanced Machine Learning
- **Ensemble Methods**: XGBoost, Random Forest, Gradient Boosting
- **Model Selection**: Systematic comparison of 6 classification algorithms
- **Hyperparameter Optimization**: Random Grid Search with stratified cross-validation
- **Feature Selection**: Recursive Feature Elimination (RFE) and Permutation Feature Importance

### 2. Production-Ready Code Architecture
```
MetalComplexClassifier/
├── decisionanalyzer/          # Core ML library (modular design)
│   ├── __init__.py
│   ├── classifier.py          # Model training and evaluation
│   ├── tree_analyzer.py       # Decision tree interpretation
│   └── utils.py               # Feature engineering utilities
├── data/                      # Dataset management
├── models/                    # Trained model artifacts
└── notebooks/                 # Research and analysis
```

### 3. Scientific Computing Stack
- **Cheminformatics**: RDKit for molecular descriptor computation
- **Data Processing**: Pandas, NumPy for efficient large-scale data manipulation
- **Visualization**: Matplotlib, Seaborn for publication-quality figures
- **Model Persistence**: Joblib for serialization

### 4. Interpretable AI
- Custom decision path analysis revealing molecular structure-activity relationships
- Feature importance ranking identifying key chemical descriptors
- Pure leaf node extraction for rule-based chemical design

---

## 🔬 Scientific Impact

**Published Research**: Journal of Cheminformatics (2025)  
**Citation**: Vigna et al., "Prediction of Pt, Ir, Ru, and Rh complexes light absorption in the therapeutic window for phototherapy using machine learning"

### Innovation
This represents the **first application of interpretable ML** to predict UV-vis absorption properties of metal complexes for phototherapy, bridging the gap between computational chemistry and applied AI.

---

## 🛠️ Installation & Usage

### Quick Start
```bash
git clone https://github.com/vorsamaqoy/MetalComplexClassifier.git
cd MetalComplexClassifier
pip install -r requirements.txt
```

### Basic Usage
```python
import decisionanalyzer as dan

analyzer = dan.TreeAnalyzer(model, X_train, y_train)

analyzer.get_pure_leaves()
analyzer.sample_decision_path(sample_id=1770)
analyzer.visualize_tree_rules()
```

### Training Pipeline
```python
from decisionanalyzer import ModelPipeline

pipeline = ModelPipeline()
pipeline.load_data('data/complexes.csv')
pipeline.preprocess()
pipeline.train_xgboost()
pipeline.evaluate()
```

---

## 📊 Key Features

### 1. Automated Feature Engineering
Computes 118 molecular descriptors from SMILES notation:
- Topological distances (T descriptors)
- Functional group counts (F descriptors)
- Bond connectivity patterns (B descriptors)

### 2. Model Interpretability Tools
```python
analyzer.get_feature_importance()

pure_leaves = analyzer.get_pure_leaves()

decision_rules = analyzer.extract_rules(node_id=780)
```

### 3. Chemical Design Support
Extract decision paths for inactive compounds to identify structural modifications:
```python
inactive_sample = 1770
analyzer.suggest_modifications(inactive_sample)
```

---

## 📈 Results & Validation

### Cross-Validation Strategy
- **Stratified 10-Fold CV** maintaining class distribution
- **Train/Test Split**: 80/20 with random state for reproducibility
- **Metrics**: Comprehensive evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)

### Comparative Analysis
Benchmark against 5 baseline algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier
- Gradient Boosting Classifier

**XGBoost + AtomPairs2D emerged as the optimal combination**

---

## 🎓 Applications

### Pharmaceutical Industry
- **High-throughput screening** of photoactive drug candidates
- **Lead optimization** for PDT/PACT agents
- **Reduce R&D costs** by eliminating unpromising compounds early

### Computational Chemistry
- **Faster than TD-DFT** by orders of magnitude
- **Resource-efficient** alternative for large-scale predictions
- **Interpretable results** for rational molecular design

---

## 🔧 Technical Stack

| Category | Technologies |
|----------|-------------|
| **ML Frameworks** | XGBoost, Scikit-learn, Pandas |
| **Molecular Descriptors** | RDKit, AtomPairs2D, WalkAndPathCounts |
| **Data Science** | NumPy, SciPy, Matplotlib, Seaborn |
| **Development** | Python 3.10+, Jupyter, Git |
| **Research Tools** | Gaussian16 (TD-DFT validation) |

---

## 🏆 Why This Project Stands Out

### Business Value
✅ **Production-Ready**: Modular, tested, deployable code  
✅ **Scalable**: Handles datasets of 10K+ compounds efficiently  
✅ **Documented**: Clear code architecture and API  
✅ **Validated**: Peer-reviewed scientific methodology  

### Technical Excellence
✅ **Modern ML Pipeline**: From raw data to deployed model  
✅ **Best Practices**: Cross-validation, feature selection, hyperparameter tuning  
✅ **Interpretability**: Actionable insights, not just black-box predictions  
✅ **Domain Integration**: Chemistry + AI in production environment  

---

## 📚 Citation

```bibtex
@article{vigna2025metalcomplex,
  title={Prediction of Pt, Ir, Ru, and Rh complexes light absorption in the therapeutic window for phototherapy using machine learning},
  author={Vigna, V. and Cova, T.F.G.G. and Pais, A.A.C.C. and Sicilia, E.},
  journal={Journal of Cheminformatics},
  volume={17},
  number={1},
  year={2025},
  publisher={Springer}
}
```

---

## 👤 Author

**Vincenzo Vigna**  
Computational Chemistry & Machine Learning Researcher

Transitioning from academic research to AI/ML engineering roles in the tech industry.

**Core Expertise**: Applied Machine Learning • Cheminformatics • Python Development • Production ML Systems

---

## 📫 Connect

[![GitHub](https://img.shields.io/badge/GitHub-vorsamaqoy-black?logo=github)](https://github.com/vorsamaqoy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:vincenzovigna@unical.it)

---

## 📄 License

This project is licensed under the terms specified in the LICENSE file. Research data available upon reasonable request.
