import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data_path = "Dataset (1).csv"
df = pd.read_csv(data_path, encoding='latin-1') # Sometimes files from certain sources use latin-1

# Display basic info
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Feature Selection
# Choosing features that likely affect ratings
features = ['Country Code', 'City', 'Cuisines', 'Average Cost for two', 
            'Has Table booking', 'Has Online delivery', 'Price range', 'Votes']
target = 'Aggregate rating'

X = df[features].copy()
y = df[target].copy()

# Handing missing values in Cuisines (small number expected)
X['Cuisines'] = X['Cuisines'].fillna('Unknown')

# Encoding Categorical Variables
# For City and Cuisines, since there are many unique values, we'll use Label Encoding for simplicity 
# though for high cardinality, other methods might be better.
le = LabelEncoder()
X['City'] = le.fit_transform(X['City'])
X['Cuisines'] = le.fit_transform(X['Cuisines'])

# Binary encoding for Yes/No columns
X['Has Table booking'] = X['Has Table booking'].map({'Yes': 1, 'No': 0})
X['Has Online delivery'] = X['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 2: Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    return mse, r2

evaluate(y_test, lr_pred, "Linear Regression")
evaluate(y_test, dt_pred, "Decision Tree Regressor")

# Feature Importance (Decision Tree)
importances = dt_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Decision Tree):")
print(feature_importance_df)

# Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Restaurant Ratings')
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as feature_importance.png")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, dt_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings (Decision Tree)')
plt.savefig('actual_vs_predicted.png')
print("Actual vs Predicted plot saved as actual_vs_predicted.png")
