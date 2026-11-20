# Analysis: ML & Deep Learning Case Studies

A collection of end-to-end machine learning projects: data exploration, preprocessing, modelling and evaluation in Jupyter notebooks.

## Projects

### 🧠 Mental Health Text Classification
📁 [`Mental_Health_Classification/`](Mental_Health_Classification)

Classifies a person's mental-health status from written text.

- Text cleaning with NLTK: lowercasing, stop-word removal, tokenisation
- EDA: class distribution, word-count distribution, word clouds
- **Hybrid model:** DistilBERT transformer embeddings fed into 1D-CNN layers
- Evaluation: accuracy, macro F1, ROC-AUC, confusion matrix

**Stack:** PyTorch · Hugging Face Transformers · TensorFlow/Keras · scikit-learn · NLTK · Plotly

### 🎓 Student Dropout Risk Prediction
📁 [`Student Drop out risk/`](Student%20Drop%20out%20risk)

Predicts which students are at risk of dropping out from academic and personal features.

- Dataset overview, cleaning and EDA
- Models: Logistic Regression, Random Forest, XGBoost, CatBoost, MLP
- **Stacking ensemble** that combines the best models
- **Explainability** with SHAP, showing which features drive the risk

**Stack:** scikit-learn · XGBoost · CatBoost · SHAP · pandas · seaborn

## How to run

```bash
git clone https://github.com/ovibaridar/Analysis-.git
cd Analysis-
pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost catboost shap nltk torch transformers tensorflow
jupyter notebook
```

Then open the notebook inside each project folder.
