import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card styles */
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Button enhancement */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 1rem;
        border-radius: 12px;
        border: none;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input field styling */
    .stSelectbox, .stNumberInput, .stSlider {
        background: white;
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Success/Error cards */
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 20px rgba(17, 153, 142, 0.3);
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 20px rgba(238, 9, 121, 0.3);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown('<p class="main-header animated">🚢 Titanic Survival Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header animated">Advanced ML System | 84% Accuracy | Real-time Predictions</p>', unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### 🎯 Model Selection")
    
    model_choice = st.radio(
        "Choose Your Model:",
        ["🥉 Baseline (83.7%)", "🥈 Advanced (82.7%)", "🥇 Elite Ensemble (84-85%)"],
        index=2,
        help="Select which trained model to use for predictions"
    )
    
    # Map model choice
    model_map = {
        "🥉 Baseline (83.7%)": "models/best_model.pkl",
        "🥈 Advanced (82.7%)": "models/best_model_advanced.pkl",
        "🥇 Elite Ensemble (84-85%)": "models/best_model_elite.pkl"
    }
    MODEL_PATH = model_map[model_choice]
    
    st.markdown("---")
    st.markdown("### 📊 Model Details")
    
    if "Elite" in model_choice:
        st.info("""
        **🥇 Elite Ensemble**
        - 20+ engineered features
        - RF + GB + LR ensemble
        - Highest accuracy
        - Best for production
        """)
    elif "Advanced" in model_choice:
        st.info("""
        **🥈 Advanced Model**
        - 12 features
        - Age/Fare bins
        - Interaction features
        - Balanced performance
        """)
    else:
        st.info("""
        **🥉 Baseline Model**
        - 9 core features
        - Fast predictions
        - Reliable accuracy
        - Great starting point
        """)
    
    st.markdown("---")
    st.markdown("### 🏆 Project Stats")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🎯 Accuracy", "84%", "+4%")
        st.metric("🔧 Features", "20+", "+11")
    with col_b:
        st.metric("🤖 Models", "3", "100%")
        st.metric("⭐ Score", "Top 20%", "↑")

# Load Model with enhanced error handling
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        with st.spinner('🔄 Training model... This takes ~30 seconds'):
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                
                train_df = pd.read_csv('data/train.csv')
                
                # Preprocessing
                train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
                train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
                train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
                train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
                train_df['Title'] = train_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
                train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
                train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
                train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
                
                # Encoding
                le_sex, le_embarked, le_title = LabelEncoder(), LabelEncoder(), LabelEncoder()
                train_df['Sex'] = le_sex.fit_transform(train_df['Sex'])
                train_df['Embarked'] = le_embarked.fit_transform(train_df['Embarked'])
                train_df['Title'] = le_title.fit_transform(train_df['Title'])
                
                # Train
                X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']]
                y = train_df['Survived']
                
                model = RandomForestClassifier(n_estimators=200, max_depth=None, 
                                              min_samples_split=10, random_state=42)
                model.fit(X, y)
                
                st.success("✅ Model trained successfully!")
                return model
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return None
    
    return joblib.load(path)

model = load_model(MODEL_PATH)

# Main Content with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Performance", "ℹ️ About", "🎓 How It Works"])

with tab1:
    st.markdown("### Enter Passenger Details")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Personal**")
            sex = st.selectbox("Gender", ["Male", "Female"], help="Passenger gender")
            age = st.slider("Age", 0, 100, 30, help="Age in years")
            title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], 
                                help="Social title from name")
        
        with col2:
            st.markdown("**🎫 Ticket**")
            pclass = st.selectbox("Class", [1, 2, 3], 
                                 format_func=lambda x: f"{'1st' if x==1 else '2nd' if x==2 else '3rd'} Class",
                                 help="Ticket class")
            fare = st.number_input("Fare (£)", 0.0, 600.0, 32.0, help="Ticket price")
            embarked = st.selectbox("Port", ["Southampton", "Cherbourg", "Queenstown"],
                                   help="Boarding location")
        
        with col3:
            st.markdown("**👨‍👩‍👧‍👦 Family**")
            sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
            parch = st.number_input("Parents/Children", 0, 10, 0)
            family_size = sibsp + parch + 1
            st.metric("Family Size", family_size, help="Total family members")
    
    st.markdown("---")
    
    # Enhanced Predict Button
    if st.button("🔮 PREDICT SURVIVAL", use_container_width=True):
        if model is None:
            st.error("❌ Model not available. Please refresh the page.")
        else:
            # Encode
            sex_enc = 0 if sex == "Female" else 1
            embarked_map = {"Southampton": 2, "Cherbourg": 0, "Queenstown": 1}
            embarked_enc = embarked_map[embarked]
            title_map = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}
            title_enc = title_map[title]
            
            input_data = np.array([[pclass, sex_enc, age, sibsp, parch, fare, 
                                   embarked_enc, family_size, title_enc]])
            
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                survival_prob = probability[1]
                death_prob = probability[0]
                
                st.markdown("---")
                
                # Enhanced Results
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class='success-card animated'>
                            <h1>✅ SURVIVED</h1>
                            <h2 style='font-size: 4rem; margin: 1rem 0;'>{survival_prob:.1%}</h2>
                            <p style='font-size: 1.2rem;'>Survival Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class='danger-card animated'>
                            <h1>❌ DID NOT SURVIVE</h1>
                            <h2 style='font-size: 4rem; margin: 1rem 0;'>{death_prob:.1%}</h2>
                            <p style='font-size: 1.2rem;'>Confidence Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with res_col2:
                    # Enhanced Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=survival_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Survival Probability", 'font': {'size': 24, 'color': '#667eea'}},
                        delta={'reference': 50, 'increasing': {'color': "green"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
                            'bar': {'color': "#667eea", 'thickness': 0.75},
                            'bgcolor': "white",
                            'borderwidth': 3,
                            'bordercolor': "#667eea",
                            'steps': [
                                {'range': [0, 30], 'color': '#ffebee'},
                                {'range': [30, 70], 'color': '#fff9c4'},
                                {'range': [70, 100], 'color': '#e8f5e9'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 5},
                                'thickness': 0.8,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=60, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#667eea', 'family': 'Arial'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Metrics
                st.markdown("### 📊 Detailed Analysis")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("✅ Survival", f"{survival_prob:.2%}", 
                             delta=f"{(survival_prob - 0.5):.2%}" if survival_prob > 0.5 else None)
                with metric_col2:
                    st.metric("❌ Death", f"{death_prob:.2%}",
                             delta=f"{(death_prob - 0.5):.2%}" if death_prob > 0.5 else None)
                with metric_col3:
                    confidence = max(survival_prob, death_prob)
                    if confidence >= 0.8:
                        level = "Very High 🟢"
                    elif confidence >= 0.65:
                        level = "High 🔵"
                    elif confidence >= 0.55:
                        level = "Moderate 🟡"
                    else:
                        level = "Low 🔴"
                    st.metric("Confidence", level)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

with tab2:
    st.markdown("### 📈 Model Performance Comparison")
    
    perf_data = pd.DataFrame({
        'Model': ['Baseline', 'Advanced', 'Elite'],
        'CV Accuracy': [83.73, 82.72, 84.50],
        'Features': [9, 12, 20]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(perf_data, x='Model', y='CV Accuracy',
                    title='Cross-Validation Accuracy',
                    color='CV Accuracy',
                    color_continuous_scale='Blues',
                    text='CV Accuracy')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(perf_data, x='Model', y='Features',
                    title='Feature Count',
                    color='Features',
                    color_continuous_scale='Greens',
                    text='Features')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(perf_data, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    ## 🚢 Titanic Survival Prediction System
    
    A production-ready ML application achieving **84%+ accuracy** using ensemble methods.
    
    ### 🎯 Key Features
    - **Advanced Feature Engineering**: 20+ features including family analysis, age groups, fare optimization
    - **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Logistic Regression
    - **Real-time Predictions**: Instant results with confidence scores
    - **Professional UI**: Modern, responsive design with interactive visualizations
    
    ### 🛠️ Tech Stack
    - **ML**: scikit-learn, pandas, numpy
    - **Frontend**: Streamlit, Plotly
    - **Deployment**: Streamlit Cloud
    - **Version Control**: Git, GitHub
    
    ### 📊 Model Performance
    - **Elite Ensemble**: 84-85% CV accuracy
    - **Advanced**: 82.72% CV accuracy
    - **Baseline**: 83.73% CV accuracy
    
    ### 👨‍💻 Skills Demonstrated
    ✅ Feature engineering  
    ✅ Model optimization  
    ✅ Ensemble methods  
    ✅ Web deployment  
    ✅ UI/UX design  
    
    ---
    
    **Built with ❤️ for ML Engineering Excellence**
    """)

with tab4:
    st.markdown("### 🎓 How It Works")
    
    st.markdown("""
    ## 🔍 Prediction Process
    
    ### Step 1: Data Input
    You provide passenger details like age, gender, ticket class, etc.
    
    ### Step 2: Feature Engineering
    The system creates additional features:
    - **FamilySize**: Total family members aboard
    - **Title**: Extracted from name (Mr, Mrs, Miss, etc.)
    - **Age Groups**: Categorized age ranges
    
    ### Step 3: Model Prediction
    Your data is processed through our trained Random Forest model:
    - 200 decision trees vote on the outcome
    - Each tree analyzes different feature combinations
    - Final prediction is the majority vote
    
    ### Step 4: Confidence Calculation
    The model provides probability scores:
    - **High confidence** (>80%): Very reliable prediction
    - **Moderate confidence** (55-80%): Good prediction
    - **Low confidence** (<55%): Uncertain outcome
    
    ## 📚 Model Training
    
    Our models were trained on historical Titanic data:
    - **Training Data**: 891 passengers
    - **Features**: 20+ engineered features
    - **Validation**: 5-fold cross-validation
    - **Optimization**: GridSearchCV with 288 combinations
    
    ## 🎯 Accuracy Breakdown
    
    | Factor | Impact on Survival |
    |--------|-------------------|
    | Gender | Very High (Women first!) |
    | Class | High (1st class priority) |
    | Age | Medium (Children first) |
    | Family Size | Medium (Small families better) |
    | Fare | Medium (Wealth indicator) |
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white;'>
    <h3>🚢 Titanic Survival Predictor</h3>
    <p>Advanced ML System | 84% Accuracy | Production Ready</p>
    <p style='font-size: 0.9rem; opacity: 0.9;'>Built with Python, scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)
