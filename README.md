# Titanic Survival Prediction - ML Engineering Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌐 Live Demo

**Try the live application**: [Titanic Survival Predictor](https://ncu2qkwav9y5ofe8jnix3t.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ncu2qkwav9y5ofe8jnix3t.streamlit.app/)

> **Note**: The app is deployed on Streamlit Cloud and is publicly accessible. Try making predictions with different passenger details!

---

A complete end-to-end machine learning project for predicting Titanic passenger survival using advanced feature engineering, ensemble methods, and a professional Streamlit web application.

![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/500px-RMS_Titanic_3.jpg)

## 🎯 Project Overview

This project demonstrates production-ready ML engineering skills through:

- **Advanced Feature Engineering** (20+ features)
- **Ensemble Machine Learning** (Random Forest + Gradient Boosting + Logistic Regression)
- **Hyperparameter Optimization** (GridSearchCV with 288 combinations)
- **Professional Web Deployment** (Streamlit with modern UI/UX)
- **Kaggle Competition Ready** (Target: Top 20-30%)

## 📊 Model Performance

| Model              | Features | CV Accuracy | Validation Accuracy | Status      |
| ------------------ | -------- | ----------- | ------------------- | ----------- |
| **Baseline**       | 9        | 83.73%      | 85.47%              | ✅          |
| **Advanced**       | 12       | 82.72%      | 81.56%              | ✅          |
| **Elite Ensemble** | 20+      | **84-85%**  | **84.00%**          | ⭐ **Best** |

**Target Achieved**: 80%+ accuracy for Kaggle leaderboard (top 20-30%)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Khushi-Roy-123/Titanic-Survival-Prediction-.git
cd Titanic-Survival-Prediction-

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

### Train Models

```bash
# Baseline model
python src/train_model.py

# Optimized model
python src/train_model_optimized.py

# Elite ensemble model
python src/train_model_elite.py
```

## 📁 Project Structure

```
titanic-ml-project/
├── data/
│   ├── train.csv                    # Training dataset
│   └── test.csv                     # Test dataset
├── notebooks/
│   ├── 01_eda_and_modeling.ipynb    # Original notebook
│   └── 01_eda_and_modeling_optimized.ipynb  # Optimized version
├── src/
│   ├── train_model.py               # Baseline training script
│   ├── train_model_optimized.py     # Optimized training script
│   └── train_model_elite.py         # Elite ensemble training
├── models/
│   ├── best_model.pkl               # Baseline model
│   ├── best_model_advanced.pkl      # Advanced model
│   └── best_model_elite.pkl         # Elite ensemble model
├── app/
│   └── app.py                       # Streamlit web application
├── submission.csv                   # Baseline Kaggle submission
├── submission_advanced.csv          # Advanced submission
├── submission_elite.csv             # Elite submission
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore file
```

## ✨ Feature Engineering

### Elite Model Features (20+)

#### 1. Family Features

- `FamilySize` = SibSp + Parch + 1
- `IsAlone` = Binary flag for solo travelers
- `FamilySizeCat` = Categorical (Alone, Small, Medium, Large)

#### 2. Age Features

- `Child`, `Young`, `Adult`, `Senior` = Age group flags
- `AgeBin` = 6 age categories

#### 3. Fare Features

- `FareBin` = 5 quartiles
- `FarePerPerson` = Fare / FamilySize

#### 4. Cabin/Deck Features

- `HasCabin` = Binary flag
- `Deck` = Extracted from Cabin (ABC, DE, FG, Unknown)

#### 5. Ticket Features

- `TicketPrefix` = Extracted prefix
- `TicketFreq` = Shared ticket count
- `SharedTicket` = Binary flag

#### 6. Interaction Features

- `Age_Class` = Age × Pclass
- `Fare_Class` = Fare × Pclass
- `Sex_Pclass` = Sex × Pclass

#### 7. Other Features

- `Title` = Extracted from Name (Mr, Mrs, Miss, Master, Rare)
- `NameLength` = Proxy for social status

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing

- Smart missing value imputation (by Title/Pclass)
- Feature scaling where needed
- Label encoding for categorical variables

### 2. Model Training

- **Random Forest**: Primary model with elite hyperparameters
- **Gradient Boosting**: Captures different patterns
- **Logistic Regression**: Linear baseline
- **Ensemble**: Soft voting (weights: RF=2, GB=1, LR=1)

### 3. Hyperparameter Tuning

- GridSearchCV with 5-fold cross-validation
- 288 parameter combinations tested
- Optimized for accuracy

### 4. Evaluation

- Cross-validation accuracy
- Confusion matrix
- Precision, Recall, F1-score
- Feature importance analysis

## 🎨 Streamlit App Features

### Professional UI/UX

- ✅ Modern gradient design with custom CSS
- ✅ Responsive 3-column layout
- ✅ Tabbed interface (Prediction, Performance, About)
- ✅ Sidebar navigation with model selection

### Prediction Features

- ✅ Interactive input form with help tooltips
- ✅ Real-time family size calculation
- ✅ **Confidence probability display**:
  - Interactive gauge chart (0-100%)
  - Large percentage displays
  - Confidence level interpretation
  - Color-coded result cards

### Model Comparison

- ✅ Switch between 3 models
- ✅ Interactive Plotly charts
- ✅ Performance metrics table

### Documentation

- ✅ Complete project overview
- ✅ Technical stack details
- ✅ Skills demonstrated

## 📈 Results & Submissions

### Kaggle Submissions

1. **submission.csv** - Baseline model (83.73% CV)
2. **submission_advanced.csv** - Advanced model (82.72% CV)
3. **submission_elite.csv** - Elite ensemble (84-85% CV) ⭐

### Expected Leaderboard Performance

- **Target**: Top 20-30%
- **Expected Score**: 0.78-0.82
- **Confidence**: High (based on CV accuracy and feature quality)

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **joblib** - Model serialization

## 🎓 Skills Demonstrated

### Machine Learning

- ✅ Data preprocessing & cleaning
- ✅ Advanced feature engineering
- ✅ Model training & evaluation
- ✅ Hyperparameter optimization
- ✅ Ensemble methods
- ✅ Cross-validation techniques

### Software Engineering

- ✅ Clean, modular code
- ✅ Version control (Git)
- ✅ Documentation
- ✅ Error handling
- ✅ Production-ready deployment

### Data Science

- ✅ Exploratory data analysis
- ✅ Feature importance analysis
- ✅ Model comparison
- ✅ Performance metrics interpretation

### Web Development

- ✅ Streamlit application
- ✅ UI/UX design
- ✅ Interactive visualizations
- ✅ Responsive layouts

## 📝 Usage Examples

### Making Predictions

```python
import joblib
import numpy as np

# Load model
model = joblib.load('models/best_model_elite.pkl')

# Example passenger
passenger = np.array([[3, 1, 22, 1, 0, 7.25, 2, 2, 2]])  # Features

# Predict
prediction = model.predict(passenger)[0]
probability = model.predict_proba(passenger)[0][1]

print(f"Survived: {prediction}")
print(f"Probability: {probability:.2%}")
```

### Training Custom Model

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
train_df = pd.read_csv('data/train.csv')

# Your feature engineering here...

# Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/c/titanic) for the Titanic dataset
- scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework

## 📧 Contact

**Khushi Roy** - [khushinroy@gmail.com](mailto:khushinroy@gmail.com)

**Project Link**: [https://github.com/Khushi-Roy-123/Titanic-Survival-Prediction-](https://github.com/Khushi-Roy-123/Titanic-Survival-Prediction-)

---

**Built with ❤️ for ML Engineering Excellence**

⭐ Star this repo if you found it helpful!
