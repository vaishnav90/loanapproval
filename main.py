import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')

column_names = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
    'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class'
]
data = pd.read_csv('/Users/vaishnavanand/credit_approval/crx.data', names=column_names, na_values='?')
# print(data.head())
data['class'] = data['class'].map({'+': 1, '-': 0})
X = data.drop('class', axis=1)
y = data['class']
categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
numerical_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']


#preprocess data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# TTS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

#decision tree
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

dt_pipeline.fit(X_train, y_train)
dt_pred = dt_pipeline.predict(X_test)

print("Decision Tree Performance:")
print(classification_report(y_test, dt_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, dt_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, dt_pred):.4f}")

# random forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Performance:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, rf_pred):.4f}")

#RANDOM FOREST WITH SMOTE

rf_smote_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_smote_pipeline.fit(X_train, y_train)
rf_smote_pred = rf_smote_pipeline.predict(X_test)

print("\nRandom Forest with SMOTE Performance:")
print(classification_report(y_test, rf_smote_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_smote_pred))
print(f"Accuracy: {accuracy_score(y_test, rf_smote_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_smote_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, rf_smote_pred):.4f}")

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    rf_pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
best_pred = best_rf.predict(X_test)

print("\nTuned Random Forest Performance:")
print(classification_report(y_test, best_pred))
print("Best Parameters:", grid_search.best_params_)
print(f"Accuracy: {accuracy_score(y_test, best_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, best_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, best_pred):.4f}")

preprocessor.fit(X)
feature_names = numerical_cols + list(
    rf_pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_cols)
)
importances = best_rf.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.show()


