import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import plot_tree, export_text
from rules import HeartDiseaseExpert, Patient  # Import both HeartDiseaseExpert and Patient
import seaborn as sns
from collections import Counter

# Load the trained Decision Tree model
dt_model = joblib.load("decision_tree_model.pkl")
print("✅ Decision Tree Model Loaded Successfully!")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()  # Convert to Series
print("✅ Test Data Loaded Successfully!")

# Define feature names for explainability
feature_names = X_test.columns.tolist()

# Make predictions using the Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree Model
print("\n📌 Decision Tree Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average="weighted"))
print("Recall:", recall_score(y_test, y_pred_dt, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred_dt, average="weighted"))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# ---- Expert System Prediction ----
expert_system = HeartDiseaseExpert()

# Ensure all required fields exist in X_test for expert system
expected_columns = ["chol", "trestbps", "exang_1", "age", "fbs_1", "oldpeak", "slope_1", "ca_1", "thal_1"]
for col in expected_columns:
    if col not in X_test.columns:
        raise ValueError(f"❌ Missing required column for Expert System: {col}")

# Make predictions using the Expert System
y_pred_expert = expert_system.predict(X_test)

# Define risk mapping - assuming y_test values are already numeric (0, 1, 2)
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}

# Map expert system predictions to numeric values for evaluation
y_pred_expert_numeric = [risk_mapping.get(risk, 1) for risk in y_pred_expert]  # Default to Medium (1) for any unknown values

# Check if y_test is already numeric, if not convert it
if y_test.dtype == 'object':
    # If y_test contains string labels, map them to numeric values
    y_test_numeric = y_test.map(lambda x: risk_mapping.get(x, 1))
else:
    # If y_test is already numeric, use it as is
    y_test_numeric = y_test

# Ensure both arrays have the same length for evaluation
min_length = min(len(y_test_numeric), len(y_pred_expert_numeric))
if len(y_test_numeric) != len(y_pred_expert_numeric):
    print(f"⚠️ Warning: Length mismatch between y_test ({len(y_test_numeric)}) and y_pred_expert ({len(y_pred_expert_numeric)}). Using first {min_length} elements.")
    y_test_numeric = y_test_numeric[:min_length]
    y_pred_expert_numeric = y_pred_expert_numeric[:min_length]

# Evaluate Expert System
print("\n📌 Expert System Model Evaluation:")
print("Accuracy:", accuracy_score(y_test_numeric, y_pred_expert_numeric))
print("Precision:", precision_score(y_test_numeric, y_pred_expert_numeric, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test_numeric, y_pred_expert_numeric, average="weighted", zero_division=0))
print("F1 Score:", f1_score(y_test_numeric, y_pred_expert_numeric, average="weighted", zero_division=0))
print("Classification Report:\n", classification_report(y_test_numeric, y_pred_expert_numeric, zero_division=0))

# ---- EXPLAINABILITY ANALYSIS ----
print("\n\n🔍 EXPLAINABILITY ANALYSIS 🔍")
print("=" * 50)

# 1. Visualize the Decision Tree
try:
    plt.figure(figsize=(20, 15))
    plot_tree(dt_model, feature_names=feature_names, class_names=["Low", "Medium", "High"], 
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Model Visualization", fontsize=16)
    plt.tight_layout()
    plt.savefig("decision_tree_visualization.png")
    print("✅ Decision Tree visualization saved as 'decision_tree_visualization.png'")
except Exception as e:
    print(f"⚠️ Error visualizing decision tree: {e}")

# 2. Extract and visualize feature importance from the Decision Tree
try:
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance in Decision Tree Model', fontsize=16)
    plt.tight_layout()
    plt.savefig("decision_tree_feature_importance.png")
    print("✅ Feature importance visualization saved as 'decision_tree_feature_importance.png'")
except Exception as e:
    print(f"⚠️ Error processing feature importance: {e}")

# 3. Export textual representation of the decision tree
try:
    tree_text = export_text(dt_model, feature_names=feature_names)
    with open("decision_tree_rules.txt", "w") as f:
        f.write(tree_text)
    print("✅ Textual representation of decision tree saved as 'decision_tree_rules.txt'")
except Exception as e:
    print(f"⚠️ Error exporting decision tree text: {e}")

# 4. Analyze rule patterns in expert system without modifying the original class
# We'll use a simpler approach to analyze rule patterns based on input features and predictions

# Create a mapping of features to predictions
feature_patterns = []
for idx, row in X_test.iterrows():
    # Extract key features that are used in rules
    pattern = {
        'cholesterol': int(row['chol']),
        'blood_pressure': int(row['trestbps']),
        'smoking': "Yes" if row['exang_1'] == 1 else "No",
        'age': int(row['age']),
        'diabetes': "Yes" if row['fbs_1'] == 1 else "No",
        'bmi': float(row['oldpeak']),
        'exercise': "Regular" if row['slope_1'] == 1 else "No",
        'family_history': "Yes" if row['ca_1'] == 1 else "No",
        'stress': "High" if row['thal_1'] == 1 else "Low",
        'prediction': y_pred_expert[idx]
    }
    feature_patterns.append(pattern)

# Analyze feature patterns leading to different risk levels
risk_levels = ["Low", "Medium", "High"]
feature_analysis = {}

for risk in risk_levels:
    # Filter patterns for this risk level
    risk_patterns = [p for p in feature_patterns if p['prediction'] == risk]
    
    if not risk_patterns:
        feature_analysis[risk] = "No patients classified as this risk level"
        continue
    
    # Analyze common patterns
    feature_analysis[risk] = {
        'count': len(risk_patterns),
        'percentage': len(risk_patterns) / len(feature_patterns) * 100,
        'avg_age': sum(p['age'] for p in risk_patterns) / len(risk_patterns),
        'avg_cholesterol': sum(p['cholesterol'] for p in risk_patterns) / len(risk_patterns),
        'avg_blood_pressure': sum(p['blood_pressure'] for p in risk_patterns) / len(risk_patterns),
        'smoking_yes': sum(1 for p in risk_patterns if p['smoking'] == "Yes") / len(risk_patterns) * 100,
        'diabetes_yes': sum(1 for p in risk_patterns if p['diabetes'] == "Yes") / len(risk_patterns) * 100,
        'family_history_yes': sum(1 for p in risk_patterns if p['family_history'] == "Yes") / len(risk_patterns) * 100
    }

# Save feature pattern analysis
with open("expert_system_feature_analysis.txt", "w") as f:
    f.write("EXPERT SYSTEM FEATURE PATTERNS ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    for risk, analysis in feature_analysis.items():
        f.write(f"{risk} Risk Level Analysis:\n")
        f.write("-"*40 + "\n")
        
        if isinstance(analysis, str):
            f.write(f"{analysis}\n\n")
            continue
            
        f.write(f"Number of patients: {analysis['count']} ({analysis['percentage']:.2f}% of total)\n")
        f.write(f"Average age: {analysis['avg_age']:.1f} years\n")
        f.write(f"Average cholesterol: {analysis['avg_cholesterol']:.1f} mg/dL\n")
        f.write(f"Average blood pressure: {analysis['avg_blood_pressure']:.1f} mmHg\n")
        f.write(f"Patients who smoke: {analysis['smoking_yes']:.1f}%\n")
        f.write(f"Patients with diabetes: {analysis['diabetes_yes']:.1f}%\n")
        f.write(f"Patients with family history: {analysis['family_history_yes']:.1f}%\n\n")
        
        # Identify potential rules that might have fired
        f.write("Likely expert system rules that applied:\n")
        if analysis['avg_cholesterol'] > 240 and analysis['avg_age'] > 50:
            f.write("- High risk if cholesterol > 240 and age > 50\n")
        if analysis['avg_blood_pressure'] > 140 and analysis['smoking_yes'] > 50:
            f.write("- High risk if blood pressure > 140 and smoking is 'Yes'\n")
        if analysis['diabetes_yes'] > 50 and analysis['family_history_yes'] > 50:
            f.write("- High risk if diabetes is 'Yes' and family history is 'Yes'\n")
        if 200 <= analysis['avg_cholesterol'] <= 240 and analysis['avg_age'] > 40:
            f.write("- Medium risk if cholesterol is between 200 and 240 and age > 40\n")
        if 120 <= analysis['avg_blood_pressure'] <= 140:
            f.write("- Medium risk if blood pressure is between 120 and 140\n")
        if analysis['avg_cholesterol'] < 200:
            f.write("- Low risk if cholesterol < 200\n")
        if analysis['avg_blood_pressure'] < 120:
            f.write("- Low risk if blood pressure < 120\n")
        f.write("\n")

print("✅ Expert System feature pattern analysis saved as 'expert_system_feature_analysis.txt'")

# 5. Compare disagreements between models
disagreements = []
for idx, (dt_pred, es_pred) in enumerate(zip(y_pred_dt, y_pred_expert_numeric)):
    if dt_pred != es_pred:
        disagreements.append({
            'Index': idx,
            'Decision Tree Prediction': dt_pred,
            'Expert System Prediction': es_pred,
            'Features': {feature: X_test.iloc[idx][feature] for feature in feature_names}
        })

# Save disagreement analysis
if disagreements:
    with open("model_disagreements.txt", "w") as f:
        f.write(f"Found {len(disagreements)} disagreements between models\n")
        f.write("=" * 50 + "\n\n")
        for i, d in enumerate(disagreements[:10]):  # Show first 10 disagreements
            f.write(f"Disagreement #{i+1}\n")
            f.write(f"Decision Tree predicted: {d['Decision Tree Prediction']}\n")
            f.write(f"Expert System predicted: {d['Expert System Prediction']}\n")
            f.write("Feature values:\n")
            for feature, value in d['Features'].items():
                f.write(f"  - {feature}: {value}\n")
            f.write("\n" + "-" * 50 + "\n\n")
    print(f"✅ Found {len(disagreements)} cases where models disagree. Analysis saved as 'model_disagreements.txt'")
else:
    print("✅ No disagreements found between models")

# 6. Generate comprehensive explainability report
with open("explainability_comparison_report.txt", "w") as f:
    f.write("EXPLAINABILITY COMPARISON: DECISION TREE VS. EXPERT SYSTEM\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. APPROACH COMPARISON\n")
    f.write("-" * 70 + "\n")
    f.write("Decision Tree Model:\n")
    f.write("  - Machine learning approach that automatically derives rules from data\n")
    f.write("  - Rules are organized in a hierarchical tree structure\n")
    f.write("  - Rules are derived based on information gain/entropy\n")
    try:
        f.write("  - Top features: " + ", ".join(feature_importance['Feature'].head(5).tolist()) + "\n\n")
    except:
        f.write("  - Top features: Could not be determined\n\n")
    
    f.write("Expert System:\n")
    f.write("  - Knowledge-based approach with human-defined rules\n")
    f.write("  - Rules are organized by risk levels (High, Medium, Low)\n")
    f.write("  - Rules incorporate domain expertise and medical knowledge\n")
    f.write("  - Transparent rule-based decision making\n\n")
    
    f.write("2. INTERPRETABILITY\n")
    f.write("-" * 70 + "\n")
    f.write("Decision Tree Interpretability:\n")
    f.write("  - Visual representation shows the decision path for any prediction\n")
    f.write("  - The path from root to leaf shows exactly which features and thresholds were used\n")
    f.write("  - The depth of the tree affects interpretability (deeper trees are harder to interpret)\n")
    if hasattr(dt_model, 'max_depth'):
        f.write(f"  - Current tree depth: {dt_model.max_depth}\n\n")
    else:
        f.write("  - Current tree depth: unknown\n\n")
    
    f.write("Expert System Interpretability:\n")
    f.write("  - Rules are explicitly defined in natural language\n")
    f.write("  - Rules have clear medical interpretations\n")
    f.write("  - Each rule corresponds to specific medical knowledge about heart disease risk factors\n\n")
    
    f.write("3. PERFORMANCE COMPARISON\n")
    f.write("-" * 70 + "\n")
    f.write(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}\n")
    f.write(f"Expert System Accuracy: {accuracy_score(y_test_numeric, y_pred_expert_numeric):.4f}\n\n")
    
    f.write("4. STRENGTHS AND WEAKNESSES\n")
    f.write("-" * 70 + "\n")
    f.write("Decision Tree Strengths:\n")
    f.write("  - Automatically discovers patterns in data\n")
    f.write("  - Can find non-obvious relationships\n")
    f.write("  - Performs well when trained on sufficient data\n\n")
    
    f.write("Decision Tree Weaknesses:\n")
    f.write("  - May overfit to training data\n")
    f.write("  - Decision boundaries are axis-parallel (limited expressiveness)\n")
    f.write("  - Deep trees can be difficult to interpret\n\n")
    
    f.write("Expert System Strengths:\n")
    f.write("  - Incorporates domain knowledge directly\n")
    f.write("  - Rules have clear medical rationale\n")
    f.write("  - Can work with limited data\n\n")
    
    f.write("Expert System Weaknesses:\n")
    f.write("  - Rules are manually defined and may miss data-driven patterns\n")
    f.write("  - Requires domain expertise to create and update\n")
    f.write("  - May not adapt well to new or unusual cases\n\n")
    
    f.write("5. CONCLUSION\n")
    f.write("-" * 70 + "\n")
    f.write("The decision tree and expert system approaches offer complementary strengths in heart disease risk prediction:\n\n")
    
    if accuracy_score(y_test, y_pred_dt) > accuracy_score(y_test_numeric, y_pred_expert_numeric):
        f.write("The decision tree model shows better predictive performance, suggesting it has captured patterns in the data that the expert system rules may not have considered.\n\n")
    elif accuracy_score(y_test, y_pred_dt) < accuracy_score(y_test_numeric, y_pred_expert_numeric):
        f.write("The expert system demonstrates better predictive performance, suggesting that the domain knowledge encoded in its rules is particularly valuable for this task.\n\n")
    else:
        f.write("Both approaches show similar predictive performance, suggesting they may be capturing similar patterns despite their different methodologies.\n\n")
    
    f.write("A hybrid approach could potentially combine the strengths of both methods:\n")
    f.write("  - Use the decision tree to identify important features and thresholds\n")
    f.write("  - Refine expert system rules based on these insights\n")
    f.write("  - Incorporate rule importance from expert knowledge into tree construction\n")

print("✅ Comprehensive explainability report generated as 'explainability_comparison_report.txt'")

print("\n📊 SUMMARY OF EXPLAINABILITY ANALYSIS")
print("-" * 50)
print("1. Decision tree model visualized with feature importance")
print("2. Expert system feature patterns analyzed")
print("3. Comparison of decision paths and rule patterns")
print("4. Detailed analysis of model disagreements")
print("5. Comprehensive explainability report generated")
print("-" * 50)