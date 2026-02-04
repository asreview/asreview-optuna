# ASReview Fine-Tuning Protocol (v2)

## 1️⃣ Feature Extractors Caching

Feature matrices will be preprocessed and stored on Surf for computational efficiency.

### 1.1 TF-IDF
- **Lemmatization:** Yes / No
- **Stopword removal:** Standard / Extended / None
- **Feature selection:** Yes / No (e.g., top-k via chi2 or mutual information)
- **Ngram range:** `[1,1]`, `[1,2]`, `[1,3]`

### 1.2 Embeddings
- Dense embeddings: `mxbai`, `multilingual E5 large` (and others)
- **Normalization:** L2 / None
- **Feature reduction (trial):** PCA / UMAP / variance threshold

---

## 2️⃣ Optuna Classical Models

### 2.1 TF-IDF
- **Models:** SVM, Random Forest, Logistic Regression, Naive Bayes
- **Hyperparameters:**
  - SVM: `C`
  - RF: `n_estimators`, `max_depth`, `max_features`
  - LR: `C`
  - NB: `alpha`

### 2.2 Embeddings
- **Models:** SVM, RF, LR
- **Hyperparameters:** Same as above
- **Additional options:** Dimensionality reduction if needed

---

## 3️⃣ Optuna Next-Level Models (Ensemble & Neural Networks)

### 3.1 Gradient Boosting / Ensembles
- **Models:** XGBoost, LightGBM, CatBoost, AdaBoost
- **Input:** Best TF-IDF / embedding configurations from Step 2
- **Hyperparameters:** 
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

### 3.2 Shallow Neural Networks
- **Architecture:** 2 hidden layers
- **Input:** TF-IDF or embeddings (optional PCA/UMAP)
- **Hyperparameters:** 
  - Layer sizes, dropout, learning rate, batch size
