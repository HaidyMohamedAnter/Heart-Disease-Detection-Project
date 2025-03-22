# Heart-Disease-Detection-Project
Heart Disease Detection System

 Project Overview
This project is a comprehensive Heart Disease Detection System that combines a rule-based expert system using Experta and a machine learning model (Decision Tree Classifier using Scikit-Learn). The goal is to provide accurate predictions of heart disease risk by analyzing patient data. The project includes data preprocessing, visualization, model training, evaluation, and a comparison between the expert system and machine learning approach.

 Key Features
- Rule-Based Expert System powered by Experta.
- Decision Tree Classifier with hyperparameter tuning.
- Data preprocessing including normalization, encoding, and feature selection.
- Data visualization with statistical summaries and heatmaps.
- User-friendly interface using Streamlit for risk prediction.
- Performance comparison between the expert system and machine learning model.

 Project Structure

Heart_Disease_Detection/
│── data/                   # Contains the dataset (raw & cleaned)
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│── notebooks/              # Jupyter Notebooks for visualization & preprocessing
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│── rule_based_system/      # Rule-based system using Experta
│   ├── rules.py
│   ├── expert_system.py
│── ml_model/               # Decision Tree implementation
│   ├── train_model.py
│   ├── predict.py
│── utils/                  # Helper functions for data cleaning & processing
│   ├── data_processing.py
│── reports/                # Comparison reports and evaluation
│   ├── accuracy_comparison.md
│── ui/                     # Streamlit UI for user interaction
│   ├── app.py
│── README.md               # Documentation & setup instructions
│── requirements.txt        # List of dependencies


### Setup Instructions
1. Clone the repository:
bash
git clone <repository_url>

2. Navigate to the project directory:
bash
cd Heart_Disease_Detection

3. Install dependencies:
bash
pip install -r requirements.txt

4. Run the Streamlit App:
bash
streamlit run ui/app.py


 Dataset Details
The dataset contains features such as:
- Age, cholesterol level, blood pressure, glucose, BMI.
- Lifestyle factors like exercise frequency, smoking, alcohol consumption, and stress.
- Medical history including family history and chest pain type.

 Expert System Rules Summary
- High cholesterol and age are indicators of high risk.
- High blood pressure and smoking contribute to elevated risk.
- Low BMI and regular exercise indicate lower risk.
- Genetic and lifestyle factors are combined for risk assessment.

 Machine Learning Model
- Decision Tree Classifier trained using Scikit-Learn.
- Hyperparameter tuning for optimal performance.
- Performance metrics: Accuracy, Precision, Recall, F1-Score.
- Model saved using joblib for reuse.

### Visualizations
- Correlation Heatmap to identify important features.
- Histograms and Boxplots to analyze data distribution.
- Feature Importance Plot showing key predictors of heart disease.

 Future Improvements
- Improve model performance with additional algorithms.
- Deploy the app on a cloud platform for wider accessibility.
- Enhance the user interface and add more input parameters.
- Implement more complex rule structures in the expert system.
