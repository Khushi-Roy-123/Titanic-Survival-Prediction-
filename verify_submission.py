import pandas as pd

# Load and verify submission
df = pd.read_csv('submission.csv')

print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'\nFirst 5 rows:')
print(df.head())
print(f'\nLast 5 rows:')
print(df.tail())
print(f'\nData types:')
print(df.dtypes)
print(f'\nNull values:')
print(df.isnull().sum())
print(f'\nSurvived value counts:')
print(df['Survived'].value_counts())
print(f'\n✅ Submission file is Kaggle-ready!')
print(f'Total predictions: {len(df)}')
