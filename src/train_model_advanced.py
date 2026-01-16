import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

print("="*60)
print("ADVANCED TITANIC MODEL - ENHANCED FEATURE ENGINEERING")
print("="*60)

# 1. Load Data
print("\n[1/8] Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
combine = [train_df, test_df]

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. Advanced Feature Engineering
print("\n[2/8] Advanced Feature Engineering...")

for dataset in combine:
    # Basic missing value handling
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].dropna().median())
    
    # Feature 1: FamilySize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # Feature 2: IsAlone (powerful feature)
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Feature 3: Title extraction and grouping
    dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Feature 4: Age bins (children, young adults, adults, seniors)
    dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 20, 40, 60, 80], 
                                labels=[0, 1, 2, 3, 4])
    
    # Feature 5: Fare bins (quartiles)
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=[0, 1, 2, 3], duplicates='drop')
    
    # Feature 6: Deck from Cabin (first letter)
    dataset['Deck'] = dataset['Cabin'].str[0]
    dataset['Deck'] = dataset['Deck'].fillna('Unknown')
    # Group rare decks
    deck_mapping = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE', 
                    'F': 'FG', 'G': 'FG', 'T': 'Unknown', 'Unknown': 'Unknown'}
    dataset['Deck'] = dataset['Deck'].map(deck_mapping)
    
    # Feature 7: Age * Class interaction
    dataset['Age_Class'] = dataset['Age'] * dataset['Pclass']

print("New features created:")
print("  - FamilySize, IsAlone")
print("  - Title (grouped)")
print("  - AgeBin, FareBin")
print("  - Deck (from Cabin)")
print("  - Age_Class interaction")

# 3. Encoding
print("\n[3/8] Encoding categorical features...")
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_title = LabelEncoder()
le_deck = LabelEncoder()

for dataset in combine:
    dataset['Sex'] = le_sex.fit_transform(dataset['Sex'])
    dataset['Embarked'] = le_embarked.fit_transform(dataset['Embarked'])
    dataset['Title'] = le_title.fit_transform(dataset['Title'])
    dataset['Deck'] = le_deck.fit_transform(dataset['Deck'])
    dataset['AgeBin'] = dataset['AgeBin'].astype(int)
    dataset['FareBin'] = dataset['FareBin'].astype(int)
    
    # Drop original columns
    dataset.drop(['Cabin', 'Name', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)

# Prepare features
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

print(f"Final feature count: {X_train.shape[1]}")
print(f"Features: {list(X_train.columns)}")

# 4. Train-Validation Split
print("\n[4/8] Splitting data...")
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# 5. Baseline Models
print("\n[5/8] Training baseline models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_baseline_acc = 0
print("-" * 60)
for name, model in models.items():
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_val)
    acc = accuracy_score(Y_val, Y_pred)
    print(f"{name:25s}: {acc:.4f}")
    if acc > best_baseline_acc:
        best_baseline_acc = acc
print("-" * 60)

# 6. Advanced Random Forest Tuning
print("\n[6/8] Advanced Random Forest Hyperparameter Tuning...")
print("Testing 216 parameter combinations with 5-fold CV...")

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

rf_advanced = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_advanced,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

grid_search.fit(X_tr, Y_tr)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# 7. Evaluate Best Model
print("\n[7/8] Evaluating optimized model...")
best_rf = grid_search.best_estimator_
Y_pred_optimized = best_rf.predict(X_val)
optimized_acc = accuracy_score(Y_val, Y_pred_optimized)

print(f"\nValidation Accuracy: {optimized_acc:.4f}")
print(f"Baseline Accuracy: {best_baseline_acc:.4f}")
print(f"Improvement: {optimized_acc - best_baseline_acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(Y_val, Y_pred_optimized))

print("\nClassification Report:")
print(classification_report(Y_val, Y_pred_optimized))

# Cross-validation
cv_scores = cross_val_score(best_rf, X_train, Y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance
print("\nTop 10 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

# 8. Train on Full Data and Generate Submission
print("\n[8/8] Training on full dataset and generating submission...")
best_rf.fit(X_train, Y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(best_rf, 'models/best_model_advanced.pkl')
print("Model saved to models/best_model_advanced.pkl")

# Generate submission
Y_pred_submission = best_rf.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred_submission
})
submission.to_csv('submission_advanced.csv', index=False)
print("Submission saved to submission_advanced.csv")

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE!")
print("="*60)
print(f"Expected Kaggle Score: ~{cv_scores.mean():.4f}")
print("Upload submission_advanced.csv to Kaggle!")
