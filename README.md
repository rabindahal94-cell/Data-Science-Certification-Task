Heart Disease Prediction Application ❤️
Objective
This project aims to build and deploy a machine learning model that predicts the likelihood of a patient having heart disease based on their medical attributes. It serves as an end-to-end data science project, covering data collection, exploratory data analysis (EDA), model training, evaluation, and deployment as an interactive web application using Streamlit.

🚀 How to Run This Project Locally
Follow these instructions to get the project running on your local machine.

Prerequisites
Python 3.7+

pip (Python package installer)

Git

Step 1: Clone the Repository
Open your terminal or command prompt and clone this repository.

# Replace with your repository URL if you push it to GitHub
git clone <your-repository-url>
cd heart_disease_project

Step 2: Create a Virtual Environment (Recommended)
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Step 3: Install Required Libraries
Install all project dependencies from the requirements.txt file.

pip install -r requirements.txt

Step 4: Train the Predictive Model
Run the model_training.py script. This script will download the data, perform all data science tasks, and create a file named heart_disease_predictor.pkl.

python model_training.py

You will see the output of each step in your terminal, and plot images will be saved to your folder.

Step 5: Launch the Streamlit Web Application
Now, run the app.py script to start the web application.

streamlit run app.py

This command will automatically open a new tab in your web browser at http://localhost:8501, where you can interact with your heart disease prediction model.

📂 Project Deliverables
model_training.py: A Python script that covers the entire data science pipeline, from loading the dataset from an online source to training and evaluating a Logistic Regression model.

app.py: A Streamlit application that provides an interactive UI for users to input patient data and receive a prediction.

requirements.txt: A file listing all the necessary libraries for the project.

README.md: This documentation file.

🤖 Model Information
Algorithm: Logistic Regression

Dataset: Heart Disease UCI dataset, sourced from a public repository.

Target Variable: target (0 = No Heart Disease, 1 = Heart Disease)

Evaluation Metric: The model achieves a high accuracy in predicting heart disease, as detailed in the classification report generated during training.