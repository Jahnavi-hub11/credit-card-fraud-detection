import pandas as pd
import numpy as np
import sklearn
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv(r"C:\Users\talur\OneDrive\Desktop\fraud-detection\creditcard.csv", encoding='latin1')

print("Dataset loaded!")
print(df.shape)
print(df.head())

print("\nClass Distribution:")
print(df['Class'].value_counts())

sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

df['Hour'] = (df['Time'] // 3600) % 24
df = df.drop('Time', axis=1)

corr = df.corr()
relevant_features = corr['Class'][abs(corr['Class']) > 0.1].index

df = df[relevant_features]

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("Selected Features:", relevant_features)

X = df.drop('Class', axis=1)
y = df['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

sns.countplot(x=y_resampled)
plt.title("Class Distribution After SMOTE")
plt.show()

print("After SMOTE:")
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

final_probs = (rf_probs + xgb_probs) / 2

y_pred = (final_probs > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, final_probs))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, final_probs)

plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc_score(y_test, final_probs))
plt.legend()
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

import pickle
pickle.dump(rf_model, open("model.pkl", "wb"))

fraud_sample = df[df['Class'] == 1].iloc[0]
print("Fraud sample:")
print(fraud_sample)