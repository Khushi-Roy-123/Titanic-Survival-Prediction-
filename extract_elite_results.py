import re

print("="*70)
print("EXTRACTING ELITE MODEL RESULTS")
print("="*70)

# Read the output from a fresh run
import subprocess
result = subprocess.run(['python', 'src/train_model_elite.py'], 
                       capture_output=True, text=True, cwd='.')

output = result.stdout + result.stderr

# Extract key metrics
lines = output.split('\n')

for line in lines:
    if 'Final feature count' in line:
        print(f"\n✅ {line.strip()}")
    elif 'VALIDATION ACCURACY' in line:
        print(f"\n{line}")
    elif 'Random Forest:' in line or 'Gradient Boosting:' in line or 'Logistic Regression:' in line or 'Ensemble' in line:
        print(f"  {line.strip()}")
    elif 'Best Model:' in line:
        print(f"\n🏆 {line.strip()}")
    elif 'Mean CV Accuracy' in line:
        print(f"\n📊 {line.strip()}")
    elif 'Expected Kaggle Score' in line:
        print(f"\n🎯 {line.strip()}")
    elif 'TARGET ACHIEVED' in line or 'Close to target' in line:
        print(f"   {line.strip()}")

print("\n" + "="*70)
