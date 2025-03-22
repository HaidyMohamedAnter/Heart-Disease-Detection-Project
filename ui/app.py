import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from experta import *
from expert_system import Patient, HeartDiseaseExpert  # Import from expert_system.py

# Load the pre-trained Decision Tree model
try:
    dt_model = joblib.load("decision_tree_model.pkl")
    st.sidebar.success("✅ Decision Tree Model Loaded")
    # Get expected feature names from the model
    expected_features = dt_model.feature_names_in_
    st.sidebar.write(f"Expected Features: {expected_features}")
except FileNotFoundError:
    st.sidebar.error("❌ Decision Tree model file 'decision_tree_model.pkl' not found.")
    dt_model = None
    expected_features = []

# Load historical data for comparisons (from data_analysis.py)
try:
    historical_data = pd.read_csv("cleaned_data.csv")
    st.sidebar.success("✅ Historical Data Loaded")
except FileNotFoundError:
    st.sidebar.warning("⚠️ No historical data found. Some visualizations unavailable.")
    historical_data = None

# Function to run the expert system and get risk prediction
def predict_expert_risk(patient_data):
    engine = HeartDiseaseExpert()
    engine.reset()
    engine.declare(patient_data)
    engine.run()
    
    if not engine.risk_levels:
        return "Low"
    elif "High" in engine.risk_levels:
        return "High"
    elif "Medium" in engine.risk_levels:
        return "Medium"
    else:
        return "Low"

# Function to run the decision tree model
def predict_dt_risk(cholesterol, blood_pressure, smoking, age, diabetes, bmi, exercise, family_history, stress):
    if dt_model is None:
        return "N/A (Model not loaded)"
    
    # Base input data
    input_data = {
        'chol': cholesterol,
        'trestbps': blood_pressure,
        'exang_1': 1 if smoking == "Yes" else 0,
        'age': age,
        'fbs_1': 1 if diabetes == "Yes" else 0,
        'oldpeak': bmi,  # Using BMI as a proxy for oldpeak
        'slope_1': 1 if exercise == "Regular" else 0,
        'ca_1': 1 if family_history == "Yes" else 0,
        'thal_1': 1 if stress == "High" else 0
    }
    
    # Create a DataFrame with all expected features, filling missing ones with 0
    input_df = pd.DataFrame(columns=expected_features, index=[0])
    for feature in expected_features:
        if feature in input_data:
            input_df[feature] = [input_data[feature]]
        else:
            input_df[feature] = [0]  # Default for missing categorical features
    
    # Predict with decision tree
    prediction = dt_model.predict(input_df)[0]
    risk_mapping = {0: "Low", 1: "Medium", 2: "High"}
    return risk_mapping.get(prediction, "Unknown")

# Streamlit UI
def main():
    st.title("Heart Disease Risk Prediction")
    st.markdown("""
    Enter your health details below to assess your risk of heart disease.  
    This tool uses **two models**: an Expert System (rule-based) and a Decision Tree (machine learning).  
    Explore the Visualization Dashboard for deeper insights.  
    **Note**: This is for demonstration purposes only. Consult a doctor for a real diagnosis.
    """)

    # Create a form for user input
    with st.form(key="patient_form"):
        st.subheader("Your Health Information")

        # Numeric inputs
        cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=0, max_value=500, value=200, step=1)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=300, value=120, step=1)
        age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

        # Categorical inputs
        smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
        diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
        exercise = st.selectbox("Do you exercise regularly?", ["No", "Regular"])
        family_history = st.selectbox("Family history of heart disease?", ["No", "Yes"])
        stress = st.selectbox("Stress level", ["Low", "High"])

        # Submit button
        submit_button = st.form_submit_button(label="Calculate Risk")

    # Process the input and display results
    if submit_button:
        # Create a Patient object for the expert system
        patient_data = Patient(
            cholesterol=cholesterol,
            blood_pressure=blood_pressure,
            smoking=smoking,
            age=age,
            diabetes=diabetes,
            bmi=bmi,
            exercise=exercise,
            family_history=family_history,
            stress=stress
        )

        # Get predictions
        expert_risk = predict_expert_risk(patient_data)
        dt_risk = predict_dt_risk(cholesterol, blood_pressure, smoking, age, diabetes, bmi, exercise, family_history, stress)

        # Display results
        st.subheader("Your Risk Assessment")
        
        # Expert System Result
        st.write("**Expert System (Rule-Based)**")
        if expert_risk == "High":
            st.error(f"Risk Level: {expert_risk}")
            st.write("High risk detected. Please consult a healthcare professional.")
        elif expert_risk == "Medium":
            st.warning(f"Risk Level: {expert_risk}")
            st.write("Moderate risk. Consider lifestyle changes and medical advice.")
        else:
            st.success(f"Risk Level: {expert_risk}")
            st.write("Low risk. Keep maintaining a healthy lifestyle!")

        # Decision Tree Result
        st.write("**Decision Tree (Machine Learning)**")
        if dt_model is None:
            st.error("Decision Tree model not available.")
        elif dt_risk == "High":
            st.error(f"Risk Level: {dt_risk}")
            st.write("High risk detected. Please consult a healthcare professional.")
        elif dt_risk == "Medium":
            st.warning(f"Risk Level: {dt_risk}")
            st.write("Moderate risk. Consider lifestyle changes and medical advice.")
        else:
            st.success(f"Risk Level: {dt_risk}")
            st.write("Low risk. Keep maintaining a healthy lifestyle!")

        # Input Summary
        st.subheader("Your Input Summary")
        st.write(f"- Cholesterol: {cholesterol} mg/dL")
        st.write(f"- Blood Pressure: {blood_pressure} mmHg")
        st.write(f"- Age: {age} years")
        st.write(f"- BMI: {bmi}")
        st.write(f"- Smoking: {smoking}")
        st.write(f"- Diabetes: {diabetes}")
        st.write(f"- Exercise: {exercise}")
        st.write(f"- Family History: {family_history}")
        st.write(f"- Stress: {stress}")

        # Visualization Dashboard
        st.subheader("Visualization Dashboard")

        # 1. Bar Chart: Model Comparison
        risk_data = pd.DataFrame({
            "Model": ["Expert System", "Decision Tree"],
            "Risk": [expert_risk, dt_risk]
        })
        fig1 = px.bar(risk_data, x="Model", y="Risk", color="Risk", 
                      title="Risk Assessment Comparison", height=400,
                      category_orders={"Risk": ["Low", "Medium", "High"]})
        fig1.update_traces(hovertemplate="Model: %{x}<br>Risk: %{y}")
        st.plotly_chart(fig1)

        # 2. Pie Chart: Simulated Risk Distribution
        if historical_data is not None:
            # Simulate adding current user's Expert System risk to historical data
            risk_counts = historical_data['target'].value_counts().rename({0: "Low", 1: "High"}).to_dict()
            risk_counts["Low"] = risk_counts.get("Low", 0) + (1 if expert_risk == "Low" else 0)
            risk_counts["Medium"] = risk_counts.get("Medium", 0) + (1 if expert_risk == "Medium" else 0)
            risk_counts["High"] = risk_counts.get("High", 0) + (1 if expert_risk == "High" else 0)
            risk_df = pd.DataFrame(list(risk_counts.items()), columns=["Risk", "Count"])
            fig2 = px.pie(risk_df, names="Risk", values="Count", 
                          title="Risk Distribution (Including Your Data)", height=400)
            fig2.update_traces(hoverinfo="label+percent+value")
            st.plotly_chart(fig2)
        else:
            st.write("Pie chart unavailable without historical data.")

        # 3. Dynamic Stats: Compare to Historical Data
        if historical_data is not None:
            st.subheader("Your Stats vs. Historical Averages")
            # Check for available columns and calculate averages safely
            stats_available = []
            stats_df_data = {"Metric": [], "Your Value": [], "Average": []}
            
            if 'chol' in historical_data.columns:
                avg_chol = historical_data['chol'].mean()
                stats_df_data["Metric"].append("Cholesterol")
                stats_df_data["Your Value"].append(cholesterol)
                stats_df_data["Average"].append(avg_chol)
                stats_available.append("Cholesterol")
            if 'trestbps' in historical_data.columns:
                avg_bp = historical_data['trestbps'].mean()
                stats_df_data["Metric"].append("Blood Pressure")
                stats_df_data["Your Value"].append(blood_pressure)
                stats_df_data["Average"].append(avg_bp)
                stats_available.append("Blood Pressure")
            if 'oldpeak' in historical_data.columns:
                avg_bmi = historical_data['oldpeak'].mean()  # Proxy for BMI
                stats_df_data["Metric"].append("BMI")
                stats_df_data["Your Value"].append(bmi)
                stats_df_data["Average"].append(avg_bmi)
                stats_available.append("BMI")

            if stats_available:
                stats_df = pd.DataFrame(stats_df_data)
                fig3 = px.bar(stats_df, x="Metric", y=["Your Value", "Average"], barmode="group",
                              title="Your Values vs. Historical Averages", height=400)
                fig3.update_traces(hovertemplate="%{y:.2f}")
                st.plotly_chart(fig3)
            else:
                st.write("No comparable metrics available in historical data.")
        else:
            st.write("Comparison stats unavailable without historical data.")

        # 4. Feature Importance (Decision Tree)
        if dt_model:
            st.subheader("Health Factors Contribution")
            feature_importance = pd.DataFrame({
                'Feature': expected_features,
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig4 = px.bar(feature_importance, x='Feature', y='Importance', 
                          title="Feature Importance (Decision Tree)", height=400)
            fig4.update_traces(hovertemplate="Importance: %{y:.3f}")
            st.plotly_chart(fig4)
        else:
            st.write("Feature importance unavailable without Decision Tree model.")

if __name__ == "__main__":
    main()