import pandas as pd

print("="*70)
print("MODEL COMPARISON - BASELINE vs ADVANCED")
print("="*70)

# Load both submissions
submission_baseline = pd.read_csv('submission.csv')
submission_advanced = pd.read_csv('submission_advanced.csv')

print("\n📊 SUBMISSION FILE COMPARISON")
print("-" * 70)
print(f"Baseline submission shape: {submission_baseline.shape}")
print(f"Advanced submission shape: {submission_advanced.shape}")

# Compare predictions
merged = submission_baseline.merge(submission_advanced, on='PassengerId', suffixes=('_baseline', '_advanced'))
differences = (merged['Survived_baseline'] != merged['Survived_advanced']).sum()
agreement_pct = ((418 - differences) / 418) * 100

print(f"\nPrediction Agreement: {agreement_pct:.2f}%")
print(f"Different predictions: {differences} out of 418")

# Survival rate comparison
baseline_survival_rate = submission_baseline['Survived'].mean()
advanced_survival_rate = submission_advanced['Survived'].mean()

print(f"\nBaseline survival rate: {baseline_survival_rate:.4f}")
print(f"Advanced survival rate: {advanced_survival_rate:.4f}")

print("\n" + "="*70)
print("FEATURE ENGINEERING COMPARISON")
print("="*70)

print("\n📌 BASELINE MODEL (9 features):")
print("  - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")
print("  - FamilySize, Title")

print("\n📌 ADVANCED MODEL (12 features):")
print("  - Pclass, Sex, SibSp, Parch, Embarked")
print("  - FamilySize, IsAlone, Title")
print("  - AgeBin, FareBin, Deck, Age_Class")

print("\n✨ NEW FEATURES ADDED:")
print("  1. IsAlone - Binary flag for solo travelers")
print("  2. AgeBin - Age grouped into 5 bins (child, young adult, adult, senior)")
print("  3. FareBin - Fare grouped into quartiles")
print("  4. Deck - Extracted from Cabin (grouped)")
print("  5. Age_Class - Interaction feature (Age × Pclass)")

print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

print("\n📊 BASELINE MODEL:")
print("  Validation Accuracy: 85.47%")
print("  CV Accuracy: 83.73% (±6.06%)")

print("\n📊 ADVANCED MODEL:")
print("  Validation Accuracy: 81.56%")
print("  CV Accuracy: 82.72% (±4.17%)")

print("\n💡 ANALYSIS:")
print("  - CV accuracy is more reliable than single validation split")
print("  - Advanced model has lower variance (±4.17% vs ±6.06%)")
print("  - More stable predictions across different data splits")
print("  - Expected Kaggle score: ~0.82-0.83")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\n🎯 SUBMIT BOTH FILES TO KAGGLE:")
print("  1. submission.csv (baseline) - Higher validation accuracy")
print("  2. submission_advanced.csv - More features, lower variance")
print("\n  Compare actual Kaggle scores to determine best approach!")

print("\n" + "="*70)

# Show sample differences
print("\nSAMPLE PREDICTION DIFFERENCES (First 20):")
diff_samples = merged[merged['Survived_baseline'] != merged['Survived_advanced']].head(20)
if len(diff_samples) > 0:
    print(diff_samples[['PassengerId', 'Survived_baseline', 'Survived_advanced']])
else:
    print("No differences in first 20 predictions")
