# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("--- Starting Heart Disease Prediction Pipeline ---")

# ==============================================================================
# Step 1: Data Collection and Exploration
# ==============================================================================
print("\n[Step 1] Loading and Exploring the Dataset from URL...")
# Load the dataset from a new, reliable public URL from the TensorFlow repository
url = 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
df = pd.read_csv(url)

# Initial exploration
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nSummary Statistics:")
print(df.describe())
# Check for missing values - this dataset is clean initially
print("\nMissing Values Check:")
print(df.isnull().sum())
print("[Step 1] Completed.")

# ==============================================================================
# Step 2: Data Cleaning and Transformation
# ==============================================================================
print("\n[Step 2] Data Cleaning and Transformation...")
# The dataset is mostly clean, but the 'thal' column is an object (text) type.
# We need to convert its categorical values into numbers.

# Create a mapping for the 'thal' column's text values
thal_map = {'normal': 1, 'fixed': 2, 'reversible': 3}
df['thal'] = df['thal'].map(thal_map)

# *** FIX APPLIED HERE ***
# The mapping above might create NaN for unexpected values.
# We will fill these NaN values with the most frequent value (the mode).
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
# *** END OF FIX ***

print("Cleaned and transformed the 'thal' column.")
print("[Step 2] Completed.")


# ==============================================================================
# Step 3: Exploratory Data Analysis (EDA)
# ==============================================================================
print("\n[Step 3] Performing Exploratory Data Analysis...")

# Visualization 1: Target variable distribution (Heart Disease vs. No Heart Disease)
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of Heart Disease')
plt.xlabel('0 = No Heart Disease, 1 = Heart Disease')
plt.ylabel('Patient Count')
plt.savefig('heart_disease_distribution.png')
print("   - Saved target distribution plot to 'heart_disease_distribution.png'")

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Heart Disease Features')
plt.savefig('correlation_heatmap.png')
print("   - Saved correlation heatmap to 'correlation_heatmap.png'")
print("Key Insight: Features like 'cp', 'thalach', and 'slope' have a notable positive correlation with the target.")
print("[Step 3] Completed.")

# ==============================================================================
# Step 4 & 5: Feature Selection and Model Development
# ==============================================================================
print("\n[Step 4 & 5] Selecting Features and Developing the Model...")
# Define features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   - Training set size: {X_train.shape[0]} samples")
print(f"   - Testing set size: {X_test.shape[0]} samples")

# Algorithm Selection: Logistic Regression is a great baseline for binary classification.
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Model Training
print("\n   - Training the Logistic Regression model...")
model.fit(X_train, y_train)
print("   - Model training complete.")
print("[Step 4 & 5] Completed.")

# ==============================================================================
# Step 6: Model Evaluation
# ==============================================================================
print("\n[Step 6] Evaluating the Model...")
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate model performance using classification metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"   - Accuracy: {accuracy:.2f}")

print("\n   - Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("   - Saved confusion matrix plot to 'confusion_matrix.png'")
print("[Step 6] Completed.")

# ==============================================================================
# Step 7: Model Deployment (Saving the Model)
# ==============================================================================
print("\n[Step 7] Saving the Trained Model...")
joblib.dump(model, 'heart_disease_predictor.pkl')
print("   - Model successfully saved as 'heart_disease_predictor.pkl'")
print("[Step 7] Completed.")

print("\n--- Heart Disease Prediction Pipeline Finished Successfully! ---")

