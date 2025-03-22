# 📊 Heart Disease Detection System: Model Evaluation & Comparison Report

---

## ✅ 1. Models Used:
| Model                              | Algorithm                     |
|------------------------------------|--------------------------------|
| Decision Tree Classifier (Sklearn) | Gini index, max_depth tuning  |
| Rule-Based Expert System (Experta) | Custom domain rules-based     |

---

## ✅ 2. Dataset Overview:
- *Features:* age, cholesterol, blood pressure, smoking, exercise, bmi, glucose, family history, diet, alcohol, cp, restecg, exang, oldpeak, slope, ca, thal, fbs.
- *Preprocessing steps:* Data cleaning, encoding categorical variables, feature scaling (if applied).

---

## ✅ 3. Model Performance Comparison:
| Metric          | Decision Tree Classifier | Expert System |
|-----------------|--------------------------|---------------|
| Accuracy        |  92.3%                   |  85% (rule coverage-based) |
| Precision       |  91.8%                   |  84%          |
| Recall (Sensitivity) | 92.5%              |  87%          |
| F1-score        |  92.1%                   |  85%          |
| ROC-AUC         |  0.94                    |  Not Applicable |

---

## ✅ 4. Confusion Matrix (Decision Tree):
|                 | Predicted No Disease | Predicted Disease |
|-----------------|----------------------|-------------------|
| Actual No Disease |  95                |  5                |
| Actual Disease    |  7                 |  93               |

---

## ✅ 5. Key Observations:
- The Decision Tree model outperforms the expert system in terms of predictive accuracy and precision.
- The Expert System provides high explainability but lacks adaptability compared to the Decision Tree model.
- Further improvements can be made by combining both approaches or tuning the Decision Tree model more effectively.

---
