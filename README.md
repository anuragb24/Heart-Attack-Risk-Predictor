
# 🫀 Heart Disease Risk Prediction using Machine Learning

## 📌 Overview

Cardiovascular diseases are a major cause of death worldwide, with heart attacks being one of the most critical conditions. Early identification of patients at **high risk** can significantly improve treatment outcomes.

This project builds a **machine learning–based predictive system** to classify whether a patient is at risk of heart disease using clinical and demographic data. Multiple supervised learning algorithms are implemented, evaluated, and compared to identify the best-performing model.

---

## 📂 Dataset

* **File used:** `heart.csv` (303 rows × 11 columns after preprocessing)
* **Source:** UCI Heart Disease dataset (Kaggle variant).
* **Preprocessing:** Dropped redundant features: `oldpeak`, `slp`, `thall`.

### Features

| Feature      | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| **age**      | Age of patient (years)                                                                |
| **sex**      | Gender (0 = Female, 1 = Male)                                                         |
| **cp**       | Chest pain type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic) |
| **trtbps**   | Resting blood pressure (mmHg)                                                         |
| **chol**     | Serum cholesterol (mg/dl)                                                             |
| **fbs**      | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                                 |
| **restecg**  | Resting ECG results (0–2)                                                             |
| **thalachh** | Maximum heart rate achieved                                                           |
| **exng**     | Exercise-induced angina (1 = yes, 0 = no)                                             |
| **caa**      | Number of major vessels (0–3)                                                         |
| **output**   | Target variable → 0 = less chance, 1 = more chance of heart attack                    |

✅ **No missing values** were found in the dataset.
📊 Correlation analysis highlighted `cp`, `thalachh`, `exng`, and `caa` as the most influential predictors.

---

## 🎯 Objective

* To predict whether a patient is at risk of heart disease (**binary classification**).
* To compare multiple machine learning algorithms and evaluate their performance.
* To identify the most suitable model for robust and interpretable clinical risk assessment.

---

## ⚙️ Methodology

1. **Data Preprocessing**

   * Dropped irrelevant columns (`oldpeak`, `slp`, `thall`).
   * Split dataset: **70% training, 30% testing** (`random_state=101`).
   * Encoded target variable using `LabelEncoder`.

2. **Exploratory Data Analysis (EDA)**

   * Visualized distributions of features (age, sex, cp, chol, etc.).
   * Correlation heatmap to study feature importance.
   * Count plots & bar plots to explore target relationships.

3. **Model Development**

   * Logistic Regression
   * Decision Tree Classifier
   * Random Forest Classifier
   * K-Nearest Neighbors (KNN) → Optimized `k` using error rate analysis (best at `k=12`)
   * Support Vector Machine (SVM)
   * AdaBoost Classifier

4. **Evaluation Metrics**

   * Accuracy score
   * Confusion matrices
   * (Optional extension: precision, recall, F1-score)

---

## 🤖 Models & Results

The models were trained and tested on the dataset. Below are the **test set accuracies**:

| Model                      | Test Accuracy (%) |
| -------------------------- | ----------------- |
| Logistic Regression        | **85.71%**        |
| K-Nearest Neighbors (k=12) | **84.62%**        |
| Random Forest              | 80.22%            |
| Support Vector Machine     | 80.22%            |
| Decision Tree              | 69.23%            |
| AdaBoost                   | 51.65%            |

🔹 **Best models**: Logistic Regression and KNN achieved the highest performance.
🔹 Random Forest and SVM were competitive but slightly less accurate.
🔹 Decision Tree and AdaBoost underperformed due to overfitting/poor parameter defaults.

---

## 📊 Key Insights

* Logistic Regression performed best, making it **interpretable and clinically useful**.
* KNN also gave high accuracy after parameter tuning.
* Features like **chest pain type, maximum heart rate, exercise-induced angina, and number of vessels** had the strongest impact on prediction.
* Scaling was imported (`StandardScaler`) but not applied — using scaling could further improve KNN and SVM performance.

---

## ✅ Conclusion

This project demonstrates how **machine learning can be applied to healthcare data** for early detection of heart attack risk.

* **Logistic Regression (85.7%)** and **KNN (84.6%)** emerged as the top models.
* Ensemble methods like Random Forest gave stable but slightly lower accuracy.
* AdaBoost and Decision Tree struggled without tuning.
* Future work can focus on **feature scaling, hyperparameter optimization, and cross-validation** to improve performance.

Such predictive systems can serve as **decision-support tools** for doctors, helping to flag high-risk patients for preventive care.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** NumPy, Pandas, Seaborn, Matplotlib, scikit-learn
* **Environment:** Jupyter / Google Colab

---

## 🚀 How to Run

1. Clone the repository.

   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook in Jupyter/Colab.

---

 

---

✨ *This project showcases end-to-end data preprocessing, EDA, model building, and evaluation — a complete ML workflow applied to healthcare.*

---

Would you like me to also **write a shorter 3-section version** of this README (Overview → Methodology → Results & Conclusion) that’s more compact for a recruiter-friendly GitHub profile?
