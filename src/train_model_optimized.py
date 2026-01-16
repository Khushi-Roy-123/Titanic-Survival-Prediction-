import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
combine = [train_df, test_df]

# 2. EDA: Check missing
print("\nMissing values in Train:")
print(train_df.isnull().sum())

# 3. Preprocessing & Feature Engineering
print("\nPreprocessing...")
for dataset in combine:
    # Handle Missing
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    dataset['Fare'] = dataset['Fare'].fillna(test_df['Fare'].dropna().median())
    
    # Feature Engineering
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # Extract Title
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Encoding
print("Encoding categorical features...")
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_title = LabelEncoder()

for dataset in combine:
    dataset['Sex'] = le_sex.fit_transform(dataset['Sex'])
    dataset['Embarked'] = le_embarked.fit_transform(dataset['Embarked'])
    dataset['Title'] = le_title.fit_transform(dataset['Title'])
    
    # Drop Cabin, Name, Ticket, PassengerId (keep PassengerId for submission in test)
    dataset.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Prepare for Training
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
Y_train = train_df['Survived']
X_test  = test_df.drop('PassengerId', axis=1).copy()

print(f"\nTraining Shape: {X_train.shape}")

# Model Training & Evaluation
print("\nTraining baseline models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_acc = 0
results = {}

# Split for validation
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

for name, model in models.items():
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_val)
    acc = accuracy_score(Y_val, Y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\nBest Baseline Model: Random Forest with Accuracy: {best_acc:.4f}")

# Optimize Random Forest with GridSearchCV
print("\n" + "="*50)
print("Optimizing Random Forest with GridSearchCV...")
print("="*50)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_optimized = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_optimized,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

grid_search.fit(X_tr, Y_tr)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Evaluate on validation set
best_rf = grid_search.best_estimator_
Y_pred_optimized = best_rf.predict(X_val)
optimized_acc = accuracy_score(Y_val, Y_pred_optimized)

print(f"\nOptimized Random Forest Validation Accuracy: {optimized_acc:.4f}")
print(f"Improvement: {optimized_acc - best_acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(Y_val, Y_pred_optimized))

print("\nClassification Report:")
print(classification_report(Y_val, Y_pred_optimized))

# Cross-validation on full training set
cv_scores = cross_val_score(best_rf, X_train, Y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train Best Model on Full Data
print("\nTraining optimized model on full training set...")
best_rf.fit(X_train, Y_train)

# Save Model
os.makedirs('models', exist_ok=True)
joblib.dump(best_rf, 'models/best_model.pkl')
print("Optimized model saved to models/best_model.pkl")

# Generate Submission
Y_pred_submission = best_rf.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred_submission
})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

print("\n" + "="*50)
print("OPTIMIZATION COMPLETE!")
print("="*50)
