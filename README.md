# Property Price Category Classifier

A supervised machine learning pipeline that classifies residential flats into **Low / Medium / High** price categories using structural features (size, bedrooms, bathrooms, floor, age) and locational features (location tier, distance to city center, nearby schools, security level). Built for **CSE422 — Artificial Intelligence** at BRAC University.

## Problem

Real-estate platforms often need to auto-tag new listings with a price bracket before a final price is confirmed — for search filters, ranking, or flagging listings priced far outside their predicted bracket for review. This project treats that as a multi-class classification problem and benchmarks six models against it under real-world data conditions: ~10% missingness per column, mixed categorical/numerical features, and no guarantee the chosen features are actually predictive.

## Dataset

1,200 flat records, 12 columns (11 features + target), balanced across the three target classes (Low 33.5% / Medium 34.4% / High 32.1%). Full schema and missing-value breakdown are in the [project report](report/Flat_Price_Classification_Report.pdf).

## Approach

1. **EDA** — missing value audit, Z-score outlier screening, correlation heatmap, per-feature boxplots vs. target, class balance check
2. **Preprocessing** — duplicate removal, mean imputation (numeric), mode imputation (categorical), label encoding + binary encoding, stratified 70/30 train/test split, MinMaxScaler
3. **Modeling** — Decision Tree, KNN, Gaussian Naive Bayes, Random Forest, XGBoost, and a Keras neural network (Dense-64 → Dropout → Dense-32 → Dropout → Dense-3 softmax)
4. **Evaluation** — Accuracy, macro Precision/Recall/F1, multiclass ROC-AUC (one-vs-rest), confusion matrices, and a train/validation loss curve for the neural network

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Decision Tree** | 0.3833 | 0.3841 | 0.3831 | **0.3831** | 0.5375 |
| XGBoost | 0.3611 | 0.3597 | 0.3600 | 0.3598 | 0.5209 |
| Gaussian Naive Bayes | 0.3500 | 0.3453 | 0.3478 | 0.3452 | 0.4957 |
| Random Forest | 0.3444 | 0.3442 | 0.3436 | 0.3439 | 0.5019 |
| KNN (k=5) | 0.3167 | 0.3152 | 0.3196 | 0.3068 | 0.4684 |
| Neural Network | 0.3444 (test acc.) | — | — | — | — |

**Key finding:** every model — across four different learning paradigms — converges to roughly the same 32–38% accuracy band on a balanced 3-class problem, only marginally above the 33.3% random-guessing baseline. This matches the EDA: the correlation heatmap and boxplots both showed very weak association between the available features and the price category label. Rather than reporting a marginal accuracy as a success, this project treats the convergent low performance as the actual finding — full reasoning is in the [report](report/Flat_Price_Classification_Report.pdf).

## Repository Structure

```
├── data/
│   └── flat_price_dataset.csv          # raw dataset (1,200 rows)
├── notebook/
│   └── flat_price_classification.ipynb # full pipeline, EDA to model evaluation
├── report/
│   ├── Flat_Price_Classification_Report.docx
│   └── Flat_Price_Classification_Report.pdf   # full write-up with all figures
├── images/                             # exported plots (heatmap, confusion matrices, etc.)
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```
   jupyter notebook notebook/flat_price_classification.ipynb
   ```
4. Run all cells top to bottom — the dataset path in the first cell points to `data/flat_price_dataset.csv`.

## Technical Architecture
- **Language:** Python 3.x
- **Paradigm:** Data Science / Machine Learning pipeline (EDA → preprocessing → multi-model benchmarking)
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, TensorFlow/Keras, matplotlib, seaborn, scipy
- **Data Storage:** Flat-file CSV
- **Interface:** Jupyter Notebook

## Core Engineering Practices Demonstrated
- **Exploratory Data Analysis:** Missing-value audit, Z-score outlier detection, correlation analysis, and class-balance verification performed before any modeling decision.
- **Reproducible Preprocessing Pipeline:** Deterministic imputation, encoding, and a stratified train/test split with a fixed random seed for reproducibility.
- **Multi-Model Benchmarking:** Six models spanning tree-based, distance-based, probabilistic, ensemble, gradient-boosted, and deep learning paradigms evaluated under identical train/test conditions for a fair comparison.
- **Rigorous Evaluation Metrics:** Macro-averaged Precision/Recall/F1 (appropriate for a balanced multi-class problem) plus multiclass ROC-AUC, rather than relying on accuracy alone.
- **Honest Result Reporting:** Convergent near-baseline performance across all models is reported and analyzed as a genuine finding about the dataset's feature-target signal, rather than the weakest model being hidden or the result being overstated.

## Author
**Md. Tanvirul Islam Rifat**
- **GitHub:** [@tanvirul-islam-rifat](https://github.com/tanvirul-islam-rifat)
- **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/YourLinkedInProfile)
