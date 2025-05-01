# Paste or type your script code here:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preprocessing
dataset = dataset.fillna('Missing')

for column in dataset.columns:
    dataset[column] = dataset[column].astype(str)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for column in dataset.columns:
    dataset[column] = le.fit_transform(dataset[column])

# 2. Define Features and Target
target_column = 'Churn'

if target_column not in dataset.columns:
    raise ValueError(f"'{target_column}' column not found in dataset!")

X = dataset.drop(target_column, axis=1)
y = dataset[target_column]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Create a Single Figure with Two Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Feature Importance
importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis', ax=axes[1])
axes[1].set_title('Top 10 Important Features Influencing Churn')
axes[1].set_xlabel('Importance Score')
axes[1].set_ylabel('Feature')

# Adjust layout
plt.tight_layout()
plt.show()
