import pandas as pd

print("="*60)
print("KAGGLE TITANIC SUBMISSION FORMAT VERIFICATION")
print("="*60)

# Load submission file
try:
    submission = pd.read_csv('submission.csv')
    print("\n✅ File loaded successfully")
except Exception as e:
    print(f"\n❌ Error loading file: {e}")
    exit(1)

# Check shape
print(f"\n📊 SHAPE CHECK")
print(f"   Rows: {submission.shape[0]}")
print(f"   Columns: {submission.shape[1]}")
if submission.shape[0] == 418 and submission.shape[1] == 2:
    print("   ✅ Correct shape (418 rows, 2 columns)")
else:
    print(f"   ❌ Expected (418, 2), got {submission.shape}")

# Check column names
print(f"\n📋 COLUMN NAMES CHECK")
print(f"   Columns: {list(submission.columns)}")
expected_cols = ['PassengerId', 'Survived']
if list(submission.columns) == expected_cols:
    print(f"   ✅ Correct column names: {expected_cols}")
else:
    print(f"   ❌ Expected {expected_cols}, got {list(submission.columns)}")

# Check for missing values
print(f"\n🔍 MISSING VALUES CHECK")
missing = submission.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("   ✅ No missing values")
else:
    print(f"   ❌ Found {missing.sum()} missing values")

# Check data types
print(f"\n🔢 DATA TYPES CHECK")
print(submission.dtypes)
if submission['PassengerId'].dtype == 'int64' and submission['Survived'].dtype == 'int64':
    print("   ✅ Correct data types (both int64)")
else:
    print("   ❌ Incorrect data types")

# Check PassengerId range
print(f"\n🆔 PASSENGERID RANGE CHECK")
print(f"   Min: {submission['PassengerId'].min()}")
print(f"   Max: {submission['PassengerId'].max()}")
if submission['PassengerId'].min() == 892 and submission['PassengerId'].max() == 1309:
    print("   ✅ Correct range (892 to 1309)")
else:
    print(f"   ❌ Expected range 892-1309")

# Check Survived values
print(f"\n✔️ SURVIVED VALUES CHECK")
unique_survived = sorted(submission['Survived'].unique())
print(f"   Unique values: {unique_survived}")
value_counts = submission['Survived'].value_counts().sort_index()
print(f"   Value counts:\n{value_counts}")
if set(unique_survived) == {0, 1}:
    print("   ✅ Correct values (only 0 and 1)")
else:
    print(f"   ❌ Expected only 0 and 1, got {unique_survived}")

# Check for duplicates
print(f"\n🔄 DUPLICATE CHECK")
duplicates = submission['PassengerId'].duplicated().sum()
if duplicates == 0:
    print(f"   ✅ No duplicate PassengerIds")
else:
    print(f"   ❌ Found {duplicates} duplicate PassengerIds")

# Final summary
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

all_checks = [
    submission.shape == (418, 2),
    list(submission.columns) == ['PassengerId', 'Survived'],
    submission.isnull().sum().sum() == 0,
    submission['PassengerId'].dtype == 'int64',
    submission['Survived'].dtype == 'int64',
    submission['PassengerId'].min() == 892,
    submission['PassengerId'].max() == 1309,
    set(submission['Survived'].unique()) == {0, 1},
    submission['PassengerId'].duplicated().sum() == 0
]

if all(all_checks):
    print("✅ ALL CHECKS PASSED - SUBMISSION IS KAGGLE-READY!")
    print("\nYou can now upload submission.csv to Kaggle:")
    print("https://www.kaggle.com/c/titanic/submit")
else:
    print("❌ SOME CHECKS FAILED - PLEASE REVIEW ABOVE")

print("\n" + "="*60)

# Show sample
print("\nSAMPLE DATA (First 10 rows):")
print(submission.head(10))
print("\nSAMPLE DATA (Last 10 rows):")
print(submission.tail(10))
