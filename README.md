# Classificatore Complessi Metalli Transizione

**Predizione AI di Complessi Metallici Fotoattivi per la Fototerapia Oncologica**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)

---

## 🎯 Panoramica del Progetto

Questo repository implementa una **pipeline di machine learning di livello production** per predire le proprietà di assorbimento della luce di complessi di metalli di transizione (Pt, Ir, Ru, Rh) nella finestra terapeutica (600-850 nm), fondamentali per applicazioni di terapia fotodinamica (PDT) e chemioterapia fotoattivata (PACT).

**Risultato Chiave**: Sviluppato un modello AI interpretabile che **accelera lo screening dei candidati farmaceutici di 1000 volte** rispetto ai metodi tradizionali di chimica computazionale (TD-DFT), mantenendo un'**accuratezza del 96%** sui set di validazione.

---

## 🚀 Caratteristiche Tecniche

### Architettura di Machine Learning
- **Modello**: Classificatore Extreme Gradient Boosting (XGBoost) con ottimizzazione personalizzata degli iperparametri
- **Feature Engineering**: Implementati descrittori molecolari AtomPairs2D con 118 feature calcolate
- **Pipeline dei Dati**: Workflow automatizzato end-to-end da SMILES molecolare a predizione
- **Interpretabilità del Modello**: Analisi degli alberi decisionali per insight chimici azionabili

### Metriche di Performance
```
Accuratezza:    96%
Precisione:     94%
Recall:         93%
F1-Score:       93%
AUC-ROC:        0.96
```

### Scala del Dataset
- **9.775** complessi di metalli di transizione dal database Reaxys
- Classificazione binaria (attivo/inattivo nella finestra terapeutica)
- Gestito severo sbilanciamento delle classi usando strategie intelligenti di undersampling

---


### 1. Machine Learning
- **Metodi Ensemble**: XGBoost, Random Forest, Gradient Boosting
- **Selezione del Modello**: Confronto sistematico di 6 algoritmi di classificazione
- **Ottimizzazione degli Iperparametri**: Random Grid Search con cross-validation stratificata
- **Selezione delle Feature**: Recursive Feature Elimination (RFE) e Permutation Feature Importance

### 2. Architettura del Codice Production-Ready
```
MetalComplexClassifier/
├── decisionanalyzer/          # Libreria ML core (design modulare)
│   ├── __init__.py
│   ├── classifier.py          # Training e valutazione del modello
│   ├── tree_analyzer.py       # Interpretazione dell'albero decisionale
│   └── utils.py               # Utility per feature engineering
├── data/                      # Gestione del dataset
├── models/                    # Artefatti del modello addestrato
└── notebooks/                 # Ricerca e analisi
```

### 3. Stack di Calcolo Scientifico
- **Cheminformatica**: RDKit per il calcolo di descrittori molecolari
- **Elaborazione Dati**: Pandas, NumPy per manipolazione efficiente di dati su larga scala
- **Visualizzazione**: Matplotlib, Seaborn per figure di qualità pubblicabile
- **Persistenza del Modello**: Joblib per la serializzazione

### 4. AI Interpretabile
- Analisi personalizzata dei percorsi decisionali che rivelano relazioni struttura-attività molecolare
- Ranking dell'importanza delle feature identificando i descrittori chimici chiave
- Estrazione di nodi foglia puri per design chimico basato su regole

---

## 🔬 Impatto Scientifico

**Ricerca Pubblicata**: Journal of Cheminformatics (2025)  
**Citazione**: Vigna et al., "Prediction of Pt, Ir, Ru, and Rh complexes light absorption in the therapeutic window for phototherapy using machine learning"

### Innovazione
Questa rappresenta la **prima applicazione di ML interpretabile** per predire proprietà di assorbimento UV-vis di complessi metallici per fototerapia, colmando il divario tra chimica computazionale e AI applicata.

---

## 🛠️ Installazione e Utilizzo

### Avvio Rapido
```bash
git clone https://github.com/vorsamaqoy/MetalComplexClassifier.git
cd MetalComplexClassifier
pip install -r requirements.txt
```

### Utilizzo Base
```python
import decisionanalyzer as dan

analyzer = dan.TreeAnalyzer(model, X_train, y_train)

analyzer.get_pure_leaves()
analyzer.sample_decision_path(sample_id=1770)
analyzer.visualize_tree_rules()
```

### Pipeline di Training
```python
from decisionanalyzer import ModelPipeline

pipeline = ModelPipeline()
pipeline.load_data('data/complexes.csv')
pipeline.preprocess()
pipeline.train_xgboost()
pipeline.evaluate()
```

---

## 📊 Funzionalità Principali

### 1. Feature Engineering Automatizzato
Calcola 118 descrittori molecolari dalla notazione SMILES:
- Distanze topologiche (descrittori T)
- Conteggio di gruppi funzionali (descrittori F)
- Pattern di connettività dei legami (descrittori B)

### 2. Strumenti di Interpretabilità del Modello
```python
analyzer.get_feature_importance()

pure_leaves = analyzer.get_pure_leaves()

decision_rules = analyzer.extract_rules(node_id=780)
```

### 3. Supporto al Design Chimico
Estrai percorsi decisionali per composti inattivi per identificare modifiche strutturali:
```python
inactive_sample = 1770
analyzer.suggest_modifications(inactive_sample)
```

---

## 📈 Risultati e Validazione

### Strategia di Cross-Validation
- **Stratified 10-Fold CV** mantenendo la distribuzione delle classi
- **Split Train/Test**: 80/20 con random state per riproducibilità
- **Metriche**: Valutazione completa (Accuratezza, Precisione, Recall, F1, AUC-ROC)

### Analisi Comparativa
Benchmark rispetto a 5 algoritmi baseline:
- Regressione Logistica
- Albero Decisionale
- Random Forest
- Support Vector Classifier
- Gradient Boosting Classifier

**XGBoost + AtomPairs2D emerso come combinazione ottimale**

---

## 🎓 Applicazioni

### Industria Farmaceutica
- **Screening ad alto throughput** di candidati farmaceutici fotoattivi
- **Ottimizzazione dei lead** per agenti PDT/PACT
- **Riduzione dei costi R&D** eliminando precocemente composti non promettenti

### Chimica Computazionale
- **Più veloce del TD-DFT** di ordini di grandezza
- **Alternativa resource-efficient** per predizioni su larga scala
- **Risultati interpretabili** per design molecolare razionale

---

## 🔧 Stack Tecnologico

| Categoria | Tecnologie |
|----------|-------------|
| **Framework ML** | XGBoost, Scikit-learn, Pandas |
| **Descrittori Molecolari** | RDKit, AtomPairs2D, WalkAndPathCounts |
| **Data Science** | NumPy, SciPy, Matplotlib, Seaborn |
| **Sviluppo** | Python 3.10+, Jupyter, Git |
| **Strumenti di Ricerca** | Gaussian16 (validazione TD-DFT) |


## 📚 Citazione

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

## 👤 Autore

**Vincenzo Vigna**  
Ricercatore in Chimica Computazionale e Machine Learning

In transizione dalla ricerca accademica a ruoli di ingegneria AI/ML nell'industria tecnologica.

**Competenze Core**: Applied Machine Learning • Cheminformatica • Sviluppo Python • Sistemi ML in Produzione

---

## 📫 Contatti

[![GitHub](https://img.shields.io/badge/GitHub-vorsamaqoy-black?logo=github)](https://github.com/vorsamaqoy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connetti-blue?logo=linkedin)](https://www.linkedin.com/in/vincenzo-vigna-931a202a)
[![Email](https://img.shields.io/badge/Email-Contatto-red?logo=gmail)](mailto:vin.cenzo96@hotmail.it)

---

## 📄 Licenza

Questo progetto è concesso in licenza secondo i termini specificati nel file LICENSE. Dati di ricerca disponibili su richiesta ragionevole.
