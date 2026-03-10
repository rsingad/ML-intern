import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "Dataset (1).csv"
df = pd.read_csv(data_path, encoding='latin-1')

# Data Preprocessing
# Treating the first cuisine listed as the primary cuisine
df['Primary Cuisine'] = df['Cuisines'].astype(str).str.split(',').str[0]

# Handling 'nan' (string from type conversion)
df = df[df['Primary Cuisine'] != 'nan']

# Simplify: Keep only top 20 cuisines to avoid extreme sparsity and long output
top_cuisines = df['Primary Cuisine'].value_counts().nlargest(20).index
df_filtered = df[df['Primary Cuisine'].isin(top_cuisines)].copy()

print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size (Top 20 cuisines): {len(df_filtered)}")

# Feature Selection
features = ['Country Code', 'City', 'Average Cost for two', 
            'Has Table booking', 'Has Online delivery', 'Price range', 'Aggregate rating', 'Votes']
target = 'Primary Cuisine'

X = df_filtered[features].copy()
y = df_filtered[target].copy()

# Encoding categorical features
le_city = LabelEncoder()
X['City'] = le_city.fit_transform(X['City'])

# Binary encoding
X['Has Table booking'] = X['Has Table booking'].map({'Yes': 1, 'No': 0})
X['Has Online delivery'] = X['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Encoding target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Selection and Training
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report (Top 20 Cuisines):")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title('Confusion Matrix for Cuisine Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('cuisine_confusion_matrix.png')
print("\nConfusion matrix saved as cuisine_confusion_matrix.png")
