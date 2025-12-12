import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("survey.csv")
print("Dataset shape:", df.shape)
df.head()

# Drop unhelpful columns
df = df.drop(["Timestamp", "comments"], axis=1)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Clean gender column
def clean_gender(val):
    val = str(val).strip().lower()
    if 'male' in val or val in ['m', 'man', 'cis male']:
        return 'Male'
    elif 'female' in val or val in ['f', 'woman', 'cis female']:
        return 'Female'
    else:
        return 'Other'
df['Gender'] = df['Gender'].apply(clean_gender)

print("Unique genders:", df['Gender'].unique())

# Our target variable (what we want to predict)
target = 'treatment'  # Yes/No

# Encode target (Yes=1, No=0)
df[target] = df[target].map({'Yes': 1, 'No': 0})

# Drop ID-like or location-heavy columns
X = df.drop(columns=[target, 'Country', 'state'])
y = df[target]

cat_cols = X.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred))
print("\nRandom Forest ROC-AUC Score:", roc_auc_score(y_test, y_proba))


log_acc = accuracy_score(y_test, log_pred)
print(f"📈 Logistic Regression Accuracy: {log_acc:.3f}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Mental Health Treatment Prediction")
plt.legend()
plt.show()

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_features, y=top_features.index, palette="viridis")
plt.title("Top 10 Important Features Influencing Mental Health Treatment")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()
