import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ELITE TITANIC MODEL - PUSHING ACCURACY ABOVE 80%")
print("="*70)

# 1. Load Data
print("\n[1/9] Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Store PassengerId for later
test_ids = test_df['PassengerId'].copy()

# Combine for feature engineering
combine = [train_df, test_df]

# 2. ELITE FEATURE ENGINEERING
print("\n[2/9] Elite Feature Engineering...")

for dataset in combine:
    # === MISSING VALUE HANDLING (SMART) ===
    # Age: Fill by Title median (more accurate than overall median)
    dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Fill Age by Title median
    for title in dataset['Title'].unique():
        dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == title), 'Age'] = \
            dataset.loc[dataset['Title'] == title, 'Age'].median()
    
    # Embarked: mode
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    
    # Fare: median by Pclass
    for pclass in [1, 2, 3]:
        dataset.loc[(dataset['Fare'].isnull()) & (dataset['Pclass'] == pclass), 'Fare'] = \
            dataset.loc[dataset['Pclass'] == pclass, 'Fare'].median()
    
    # === FAMILY FEATURES ===
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)
    
    # Family size categories
    dataset['FamilySizeCat'] = 'Medium'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeCat'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeCat'] = 'Large'
    dataset.loc[(dataset['FamilySize'] >= 2) & (dataset['FamilySize'] <= 4), 'FamilySizeCat'] = 'Small'
    
    # === AGE FEATURES ===
    dataset['Child'] = (dataset['Age'] < 16).astype(int)
    dataset['Young'] = ((dataset['Age'] >= 16) & (dataset['Age'] < 32)).astype(int)
    dataset['Adult'] = ((dataset['Age'] >= 32) & (dataset['Age'] < 48)).astype(int)
    dataset['Senior'] = (dataset['Age'] >= 48).astype(int)
    
    # Age bins (more granular)
    dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 25, 35, 60, 80], 
                                labels=[0, 1, 2, 3, 4, 5])
    
    # === FARE FEATURES ===
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 5, labels=[0, 1, 2, 3, 4], duplicates='drop')
    dataset['FarePerPerson'] = dataset['Fare'] / dataset['FamilySize']
    
    # === CABIN/DECK FEATURES ===
    dataset['HasCabin'] = dataset['Cabin'].notna().astype(int)
    dataset['Deck'] = dataset['Cabin'].str[0]
    dataset['Deck'] = dataset['Deck'].fillna('Unknown')
    
    # Group decks
    deck_map = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE',
                'F': 'FG', 'G': 'FG', 'T': 'Unknown', 'Unknown': 'Unknown'}
    dataset['Deck'] = dataset['Deck'].map(deck_map)
    
    # === TICKET FEATURES ===
    dataset['TicketPrefix'] = dataset['Ticket'].str.split().str[0]
    dataset['TicketPrefix'] = dataset['TicketPrefix'].replace(r'[0-9]+', 'NUM', regex=True)
    
    # Ticket frequency (group tickets)
    ticket_counts = dataset['Ticket'].value_counts()
    dataset['TicketFreq'] = dataset['Ticket'].map(ticket_counts)
    dataset['SharedTicket'] = (dataset['TicketFreq'] > 1).astype(int)
    
    # === INTERACTION FEATURES ===
    dataset['Age_Class'] = dataset['Age'] * dataset['Pclass']
    dataset['Fare_Class'] = dataset['Fare'] * dataset['Pclass']
    dataset['Sex_Pclass'] = dataset['Sex'].map({'male': 0, 'female': 1}) * dataset['Pclass']
    
    # === NAME LENGTH (proxy for social status) ===
    dataset['NameLength'] = dataset['Name'].str.len()

print("\n✨ Features Created:")
print("  Family: FamilySize, IsAlone, FamilySizeCat")
print("  Age: Child, Young, Adult, Senior, AgeBin")
print("  Fare: FareBin, FarePerPerson")
print("  Cabin: HasCabin, Deck")
print("  Ticket: TicketPrefix, TicketFreq, SharedTicket")
print("  Interactions: Age_Class, Fare_Class, Sex_Pclass")
print("  Other: Title, NameLength")

# 3. ENCODING
print("\n[3/9] Encoding categorical features...")

# Label encode categorical features
categorical_features = ['Sex', 'Embarked', 'Title', 'Deck', 'FamilySizeCat', 'TicketPrefix']

for feature in categorical_features:
    le = LabelEncoder()
    # Fit on combined data to ensure same encoding
    combined_values = pd.concat([train_df[feature], test_df[feature]]).astype(str)
    le.fit(combined_values)
    train_df[feature] = le.transform(train_df[feature].astype(str))
    test_df[feature] = le.transform(test_df[feature].astype(str))

# Convert bins to int
for dataset in combine:
    dataset['AgeBin'] = dataset['AgeBin'].astype(int)
    dataset['FareBin'] = dataset['FareBin'].astype(int)

# Drop unnecessary columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
X_train = train_df.drop(['Survived'] + drop_cols, axis=1, errors='ignore')
Y_train = train_df['Survived']
X_test = test_df.drop(drop_cols, axis=1, errors='ignore')

print(f"\nFinal feature count: {X_train.shape[1]}")
print(f"Features: {list(X_train.columns)}")

# 4. Train-Validation Split
print("\n[4/9] Creating train-validation split...")
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

# 5. ELITE HYPERPARAMETER TUNING
print("\n[5/9] Elite Random Forest Hyperparameter Tuning...")
print("Testing 288 parameter combinations...")

param_grid_elite = {
    'n_estimators': [300, 400, 500],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced', None]
}

rf_elite = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_elite,
    param_grid=param_grid_elite,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

grid_search.fit(X_tr, Y_tr)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# 6. ENSEMBLE MODEL
print("\n[6/9] Creating Ensemble Model...")

# Train additional models
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)

# Voting Classifier
ensemble = VotingClassifier(
    estimators=[('rf', best_rf), ('gb', gb), ('lr', lr)],
    voting='soft',
    weights=[2, 1, 1]  # RF gets more weight
)

print("Training ensemble (RF + GB + LR)...")
ensemble.fit(X_tr, Y_tr)

# 7. EVALUATION
print("\n[7/9] Evaluating models...")

# Individual models
Y_pred_rf = best_rf.predict(X_val)
acc_rf = accuracy_score(Y_val, Y_pred_rf)

gb.fit(X_tr, Y_tr)
Y_pred_gb = gb.predict(X_val)
acc_gb = accuracy_score(Y_val, Y_pred_gb)

lr.fit(X_tr, Y_tr)
Y_pred_lr = lr.predict(X_val)
acc_lr = accuracy_score(Y_val, Y_pred_lr)

# Ensemble
Y_pred_ensemble = ensemble.predict(X_val)
acc_ensemble = accuracy_score(Y_val, Y_pred_ensemble)

print("\n" + "="*70)
print("VALIDATION ACCURACY COMPARISON")
print("="*70)
print(f"Random Forest:       {acc_rf:.4f}")
print(f"Gradient Boosting:   {acc_gb:.4f}")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"Ensemble (Voting):   {acc_ensemble:.4f}")
print("="*70)

# Choose best model
best_model = ensemble if acc_ensemble >= acc_rf else best_rf
best_acc = max(acc_ensemble, acc_rf)
model_name = "Ensemble" if acc_ensemble >= acc_rf else "Random Forest"

print(f"\n🏆 Best Model: {model_name} with {best_acc:.4f} accuracy")

# Cross-validation on full training set
print("\n[8/9] Cross-validation on full training set...")
cv_scores = cross_val_score(best_model, X_train, Y_train, cv=5, scoring='accuracy')
print(f"5-Fold CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(Y_val, Y_pred_ensemble if model_name == "Ensemble" else Y_pred_rf))
print("\nClassification Report:")
print(classification_report(Y_val, Y_pred_ensemble if model_name == "Ensemble" else Y_pred_rf))

# Feature Importance (from RF)
print("\n📊 Top 15 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(15).to_string(index=False))

# 9. FINAL TRAINING AND SUBMISSION
print("\n[9/9] Training on full dataset and generating submission...")
best_model.fit(X_train, Y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model_elite.pkl')
print("✅ Model saved to models/best_model_elite.pkl")

# Generate submission
Y_pred_submission = best_model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": Y_pred_submission
})
submission.to_csv('submission_elite.csv', index=False)
print("✅ Submission saved to submission_elite.csv")

print("\n" + "="*70)
print("🎯 ELITE MODEL TRAINING COMPLETE!")
print("="*70)
print(f"Expected Kaggle Score: ~{cv_scores.mean():.4f}")
print(f"Target: 80%+ accuracy")
print(f"Status: {'✅ TARGET ACHIEVED!' if cv_scores.mean() >= 0.80 else '⚠️ Close to target'}")
print("\nUpload submission_elite.csv to Kaggle!")
print("="*70)
