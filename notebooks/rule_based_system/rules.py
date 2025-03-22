from experta import *

# Define the Patient Facts
class Patient(Fact):
    """Patient information for heart disease risk assessment."""
    cholesterol = Field(int, mandatory=True)  # Cholesterol level
    blood_pressure = Field(int, mandatory=True)  # Blood pressure level
    smoking = Field(str, mandatory=True)  # "Yes" or "No"
    age = Field(int, mandatory=True)  # Age of the patient
    diabetes = Field(str, mandatory=True)  # "Yes" or "No"
    bmi = Field(float, mandatory=True)  # Body Mass Index
    exercise = Field(str, mandatory=True)  # "Regular" or "No"
    family_history = Field(str, mandatory=True)  # "Yes" or "No"
    stress = Field(str, mandatory=True)  # "High" or "Low"
    risk_level = Field(str, default="Low")  # Default risk level is "Low"


# Define the Expert System
class HeartDiseaseExpert(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.risk_levels = []  # Store risk levels for each patient

    # Rule 1: High risk if cholesterol > 240 and age > 50
    @Rule(Patient(cholesterol=P(lambda x: x > 240), age=P(lambda x: x > 50)))
    def high_risk_cholesterol_age(self):
        self.risk_levels.append("High")

    # Rule 2: High risk if blood pressure > 140 and smoking is "Yes"
    @Rule(Patient(blood_pressure=P(lambda x: x > 140), smoking="Yes"))
    def high_risk_bp_smoking(self):
        self.risk_levels.append("High")

    # Rule 3: High risk if diabetes is "Yes" and family history is "Yes"
    @Rule(Patient(diabetes="Yes", family_history="Yes"))
    def high_risk_diabetes_family(self):
        self.risk_levels.append("High")

    # Rule 4: High risk if BMI > 30 and exercise is "No"
    @Rule(Patient(bmi=P(lambda x: x > 30), exercise="No"))
    def high_risk_obesity_no_exercise(self):
        self.risk_levels.append("High")

    # Rule 5: High risk if stress is "High" and blood pressure > 140
    @Rule(Patient(stress="High", blood_pressure=P(lambda x: x > 140)))
    def high_risk_stress_bp(self):
        self.risk_levels.append("High")

    # Rule 6: Medium risk if cholesterol is between 200 and 240 and age > 40
    @Rule(Patient(cholesterol=P(lambda x: 200 <= x <= 240), age=P(lambda x: x > 40)))
    def medium_risk_cholesterol(self):
        self.risk_levels.append("Medium")

    # Rule 7: Medium risk if blood pressure is between 120 and 140 and stress is "High"
    @Rule(Patient(blood_pressure=P(lambda x: 120 <= x <= 140), stress="High"))
    def medium_risk_stress(self):
        self.risk_levels.append("Medium")

    # Rule 8: Medium risk if diabetes is "Yes" and BMI is between 25 and 30
    @Rule(Patient(diabetes="Yes", bmi=P(lambda x: 25 <= x <= 30)))
    def medium_risk_diabetes(self):
        self.risk_levels.append("Medium")

    # Rule 9: Low risk if exercise is "Regular" and BMI < 25
    @Rule(Patient(exercise="Regular", bmi=P(lambda x: x < 25)))
    def low_risk_healthy(self):
        self.risk_levels.append("Low")

    # Rule 10: Low risk if family history is "No" and cholesterol < 200
    @Rule(Patient(family_history="No", cholesterol=P(lambda x: x < 200)))
    def low_risk_no_family_history(self):
        self.risk_levels.append("Low")

    # Rule 11: Low risk if exercise is "Regular" and age < 30
    @Rule(Patient(exercise="Regular", age=P(lambda x: x < 30)))
    def low_risk_exercise_young(self):
        self.risk_levels.append("Low")

    # Rule 12: Low risk if smoking is "No", exercise is "Regular", and stress is "Low"
    @Rule(Patient(smoking="No", exercise="Regular", stress="Low"))
    def low_risk_healthy_lifestyle(self):
        self.risk_levels.append("Low")

    # Rule 13: Low risk if cholesterol < 200, blood pressure < 120, and BMI < 25
    @Rule(Patient(cholesterol=P(lambda x: x < 200), blood_pressure=P(lambda x: x < 120), bmi=P(lambda x: x < 25)))
    def low_risk_ideal_health(self):
        self.risk_levels.append("Low")

    # Rule 14: High risk if family history is "Yes" and age > 60
    @Rule(Patient(family_history="Yes", age=P(lambda x: x > 60)))
    def high_risk_family_history_age(self):
        self.risk_levels.append("High")

    # Rule 15: High risk if diabetes is "Yes", smoking is "Yes", and age > 50
    @Rule(Patient(diabetes="Yes", smoking="Yes", age=P(lambda x: x > 50)))
    def high_risk_diabetes_smoking_age(self):
        self.risk_levels.append("High")

    # Rule 16: High risk if stress is "High", smoking is "Yes", and BMI > 30
    @Rule(Patient(stress="High", smoking="Yes", bmi=P(lambda x: x > 30)))
    def high_risk_stress_smoking_obesity(self):
        self.risk_levels.append("High")

    # Rule 17: High risk if blood pressure > 160 and cholesterol > 260
    @Rule(Patient(blood_pressure=P(lambda x: x > 160), cholesterol=P(lambda x: x > 260)))
    def high_risk_extreme_bp_cholesterol(self):
        self.risk_levels.append("High")

    # Rule 18: High risk if age > 70, diabetes is "Yes", and family history is "Yes"
    @Rule(Patient(age=P(lambda x: x > 70), diabetes="Yes", family_history="Yes"))
    def high_risk_elderly_diabetes_family(self):
        self.risk_levels.append("High")

    # Predict method to evaluate risk for each patient
    def predict(self, X_test):
        """Predict heart disease risk for test data."""
        predictions = []  # Store predictions for each patient

        for _, row in X_test.iterrows():
            self.reset()
            self.risk_levels = []  # Reset risk levels for each patient

            # Declare patient facts
            self.declare(Patient(
                cholesterol=int(row['chol']),  # Cholesterol
                blood_pressure=int(row['trestbps']),  # Blood pressure
                smoking="Yes" if row['exang_1'] == 1 else "No",  # Smoking
                age=int(row['age']),  # Age
                diabetes="Yes" if row['fbs_1'] == 1 else "No",  # Diabetes
                bmi=float(row['oldpeak']),  # BMI
                exercise="Regular" if row['slope_1'] == 1 else "No",  # Exercise
                family_history="Yes" if row['ca_1'] == 1 else "No",  # Family history
                stress="High" if row['thal_1'] == 1 else "Low"  # Stress
            ))

            # Run the rules
            self.run()

            # Determine the final risk level
            if not self.risk_levels:
                predictions.append("Low")  # Default to "Low" if no rules match
            else:
                # Assign the highest risk level (High > Medium > Low)
                if "High" in self.risk_levels:
                    predictions.append("High")
                elif "Medium" in self.risk_levels:
                    predictions.append("Medium")
                else:
                    predictions.append("Low")

        return predictions  # Fixed: return the predictions list instead of undefined variable