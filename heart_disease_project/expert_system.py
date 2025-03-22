from experta import *
from rules import Patient, HeartDiseaseExpert

def get_user_input():
    while True:
        try:
            cholesterol = int(input("Enter your cholesterol level: "))
            if cholesterol < 0:
                print("Cholesterol level cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for cholesterol level.")
    
    while True:
        try:
            blood_pressure = int(input("Enter your blood pressure: "))
            if blood_pressure < 0:
                print("Blood pressure cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for blood pressure.")
    
    smoking = input("Do you smoke? (Yes/No): ").strip().capitalize()
    while smoking not in ["Yes", "No"]:
        print("Invalid input. Please enter 'Yes' or 'No'.")
        smoking = input("Do you smoke? (Yes/No): ").strip().capitalize()
    
    while True:
        try:
            age = int(input("Enter your age: "))
            if age < 0:
                print("Age cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for age.")
    
    diabetes = input("Do you have diabetes? (Yes/No): ").strip().capitalize()
    while diabetes not in ["Yes", "No"]:
        print("Invalid input. Please enter 'Yes' or 'No'.")
        diabetes = input("Do you have diabetes? (Yes/No): ").strip().capitalize()
    
    while True:
        try:
            bmi = float(input("Enter your BMI: "))
            if bmi < 0:
                print("BMI cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for BMI.")
    
    exercise = input("Do you exercise regularly? (Regular/No): ").strip().capitalize()
    while exercise not in ["Regular", "No"]:
        print("Invalid input. Please enter 'Regular' or 'No'.")
        exercise = input("Do you exercise regularly? (Regular/No): ").strip().capitalize()
    
    family_history = input("Do you have a family history of heart disease? (Yes/No): ").strip().capitalize()
    while family_history not in ["Yes", "No"]:
        print("Invalid input. Please enter 'Yes' or 'No'.")
        family_history = input("Do you have a family history of heart disease? (Yes/No): ").strip().capitalize()
    
    stress = input("How is your stress level? (High/Low): ").strip().capitalize()
    while stress not in ["High", "Low"]:
        print("Invalid input. Please enter 'High' or 'Low'.")
        stress = input("How is your stress level? (High/Low): ").strip().capitalize()
    
    return Patient(cholesterol=cholesterol, blood_pressure=blood_pressure, smoking=smoking,
                   age=age, diabetes=diabetes, bmi=bmi, exercise=exercise,
                   family_history=family_history, stress=stress)

# Run the Expert System
def main():
    engine = HeartDiseaseExpert()
    engine.reset()
    patient_data = get_user_input()
    
    # Convert Patient object to a dictionary
    patient_dict = patient_data.as_dict()
    
    engine.declare(Fact(**patient_dict))
    engine.run()
    
    # Print risk level after execution
    print(f"\nFinal Risk Assessment: {patient_dict.get('risk_level', 'Unknown')}")

if __name__ == "__main__":
    main()


