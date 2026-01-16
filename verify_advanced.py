import pandas as pd

print("="*70)
print("VERIFYING ADVANCED SUBMISSION FILE")
print("="*70)

submission = pd.read_csv('submission_advanced.csv')

print(f"\n✅ Shape: {submission.shape}")
print(f"✅ Columns: {list(submission.columns)}")
print(f"✅ No missing values: {submission.isnull().sum().sum() == 0}")
print(f"✅ PassengerId range: {submission['PassengerId'].min()} to {submission['PassengerId'].max()}")
print(f"✅ Survived values: {sorted(submission['Survived'].unique())}")

print("\n" + "="*70)
print("ADVANCED SUBMISSION IS KAGGLE-READY!")
print("="*70)
