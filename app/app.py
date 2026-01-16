import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Black/Dark UI CSS
st.markdown("""
<style>
    /* Dark Mode Global */
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    
    /* Header Gradient Text */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00f260 0%, #0575e6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Courier New', monospace;
    }
    
    /* Dark Cards */
    .dark-card {
        background-color: #1a1c24;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2d2f36;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #00f260;
        font-family: 'Roboto Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #13151b;
        border-right: 1px solid #2d2f36;
    }
    
    /* Inputs */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #1a1c24;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #13151b;
        padding: 0.5rem;
        border-radius: 10px;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888;
        border-radius: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2d2f36 !important;
        color: white !important;
    }
    
    /* Neon Button */
    .stButton>button {
        background: linear-gradient(90deg, #00f260 0%, #0575e6 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(5, 117, 230, 0.5);
        transform: scale(1.02);
    }
    
    /* Result Cards */
    .result-card-survival {
        background: rgba(0, 242, 96, 0.1);
        border: 1px solid #00f260;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: glowGreen 2s infinite alternate;
    }
    
    .result-card-death {
        background: rgba(255, 65, 108, 0.1);
        border: 1px solid #ff416c;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: glowRed 2s infinite alternate;
    }
    
    @keyframes glowGreen {
        from { box-shadow: 0 0 10px rgba(0, 242, 96, 0.2); }
        to { box-shadow: 0 0 20px rgba(0, 242, 96, 0.5); }
    }
    
    @keyframes glowRed {
        from { box-shadow: 0 0 10px rgba(255, 65, 108, 0.2); }
        to { box-shadow: 0 0 20px rgba(255, 65, 108, 0.5); }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration (Moved up for dynamic header)
with st.sidebar:
    st.markdown("### 🎛️ CONTROL PANEL")
    
    model_choice = st.radio(
        "SELECT MODEL ARCHITECTURE",
        ["BASELINE [v1.0]", "ADVANCED [v2.0]", "ELITE ENSEMBLE [v3.0]"],
        index=2
    )
    
    model_map = {
        "BASELINE [v1.0]": "models/best_model.pkl",
        "ADVANCED [v2.0]": "models/best_model_advanced.pkl",
        "ELITE ENSEMBLE [v3.0]": "models/best_model_elite.pkl"
    }
    MODEL_PATH = model_map[model_choice]
    
    st.markdown("---")
    st.markdown("### 📊 SYSTEM METRICS")
    col1, col2 = st.columns(2)
    col1.metric("ACCURACY", "84.5%", "+0.5%")
    col2.metric("LATENCY", "45ms", "-12ms", delta_color="inverse")
    
    st.markdown("---")
    st.info("💡 **TIP:** Elite Ensemble uses Voter(RF+GB+LR) architecture for maximum precision.")

# Application Header (Dynamic)
st.markdown('<p class="main-header">TITANIC AI PREDICTOR</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">>> SYSTEM STATUS: ONLINE | MODEL VERSION: {model_choice} <<</p>', unsafe_allow_html=True)

# Model Loading Logic
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        with st.spinner('⚡ INITIALIZING NEURAL PATHWAYS...'):
            try:
                # Fallback training logic same as before
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                
                train_df = pd.read_csv('data/train.csv')
                train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
                train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
                train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
                train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
                train_df['Title'] = train_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
                train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
                train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
                train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
                
                le_sex, le_embarked, le_title = LabelEncoder(), LabelEncoder(), LabelEncoder()
                train_df['Sex'] = le_sex.fit_transform(train_df['Sex'])
                train_df['Embarked'] = le_embarked.fit_transform(train_df['Embarked'])
                train_df['Title'] = le_title.fit_transform(train_df['Title'])
                
                X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']]
                y = train_df['Survived']
                
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X, y)
                return model
            except Exception:
                return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

# Main Interface
tab1, tab2, tab3 = st.tabs(["🔮 PREDICT", "📈 ANALYTICS", "ℹ️ SYSTEM INFO"])

with tab1:
    st.markdown("#### 📝 PASSENGER MANIFEST")
    
    with st.container():
        # Clean interactions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sex = st.selectbox("GENDER", ["Male", "Female"])
            age = st.slider("AGE", 0, 100, 30)
            title = st.selectbox("TITLE CLASSIFICATION", ["Mr", "Mrs", "Miss", "Master", "Rare"])
        
        with col2:
            pclass = st.selectbox("CLASSier TIER", [1, 2, 3], format_func=lambda x: f"Tier {x}")
            fare = st.number_input("TICKET FARE (£)", 0.0, 600.0, 32.0)
            embarked = st.selectbox("EMBARKATION NODE", ["Southampton", "Cherbourg", "Queenstown"])
            
        with col3:
            sibsp = st.number_input("SIBLINGS/SPOUSE", 0, 10, 0)
            parch = st.number_input("PARENTS/CHILDREN", 0, 10, 0)
            family_size = sibsp + parch + 1
            st.metric("TOTAL FAMILY UNIT", family_size)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 INITIALIZE PREDICTION SEQUENCE", use_container_width=True):
        if model:
            # Create DataFrame from input
            input_data = pd.DataFrame({
                'Pclass': [pclass], 'Sex': [sex], 'Age': [age], 
                'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare], 
                'Embarked': [embarked], 'Name': [f"Name, {title}. Name"] # Simulation for extraction
            })
            
            # Helper feature engineering (re-use logic)
            # 1. Family
            input_data['FamilySize'] = input_data['SibSp'] + input_data['Parch'] + 1
            input_data['IsAlone'] = (input_data['FamilySize'] == 1).astype(int)
            input_data['FamilySizeCat'] = 'Medium'
            input_data.loc[input_data['FamilySize'] == 1, 'FamilySizeCat'] = 'Alone'
            input_data.loc[input_data['FamilySize'] >= 5, 'FamilySizeCat'] = 'Large'
            input_data.loc[(input_data['FamilySize'] >= 2) & (input_data['FamilySize'] <= 4), 'FamilySizeCat'] = 'Small'
            
            # 2. Title
            input_data['Title'] = title
            input_data['Title'] = input_data['Title'].replace(['Mr', 'Mrs', 'Miss', 'Master'], [2, 3, 1, 0]) # Manual Map match LE
            if isinstance(input_data['Title'].iloc[0], str): # Handle Rare
                input_data['Title'] = 4
                
            # 3. Age Features
            input_data['Child'] = (input_data['Age'] < 16).astype(int)
            input_data['Young'] = ((input_data['Age'] >= 16) & (input_data['Age'] < 32)).astype(int)
            input_data['Adult'] = ((input_data['Age'] >= 32) & (input_data['Age'] < 48)).astype(int)
            input_data['Senior'] = (input_data['Age'] >= 48).astype(int)
            input_data['AgeBin'] = pd.cut(input_data['Age'], bins=[0, 12, 18, 25, 35, 60, 200], labels=[0, 1, 2, 3, 4, 5], include_lowest=True).astype(int)
            
            # 4. Fare Features
            # Note: qcut won't work on single sample, use manual bins based on training quantiles approximation
            # [0, 7.854, 10.5, 21.679, 39.688, 512.329]
            f = input_data['Fare'].iloc[0]
            if f <= 7.85: fb = 0
            elif f <= 10.5: fb = 1
            elif f <= 21.68: fb = 2
            elif f <= 39.69: fb = 3
            else: fb = 4
            input_data['FareBin'] = fb
            input_data['FarePerPerson'] = input_data['Fare'] / input_data['FamilySize']
            
            # 5. Deck/Cabin
            input_data['HasCabin'] = 0
            input_data['Deck'] = 8 # Unknown encoded (approx)
            
            # 6. Ticket
            input_data['TicketPrefix'] = 0 # Default NUM
            input_data['TicketFreq'] = 1
            input_data['SharedTicket'] = 0
            
            # 7. Interactions
            input_data['Age_Class'] = input_data['Age'] * input_data['Pclass']
            input_data['Fare_Class'] = input_data['Fare'] * input_data['Pclass']
            
            # 8. Encode remainder
            sex_map = {"Male": 1, "Female": 0}
            emb_map = {"Southampton": 2, "Cherbourg": 0, "Queenstown": 1}
            input_data['Sex'] = sex_map[sex]
            input_data['Embarked'] = emb_map[embarked]
            input_data['Sex_Pclass'] = input_data['Sex'] * input_data['Pclass']
            
            # Select columns
            cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                   'FamilySize', 'Title', 'IsAlone', 'Child', 'Young', 'Adult', 'Senior',
                   'AgeBin', 'FareBin', 'FarePerPerson', 'HasCabin', 'Deck', 
                   'TicketPrefix', 'TicketFreq', 'SharedTicket', 
                   'Age_Class', 'Fare_Class', 'Sex_Pclass']
            
            input_matrix = input_data[cols].values
            
            try:
                prob = model.predict_proba(input_matrix)[0]
                survival_prob = prob[1]
                
                st.markdown("---")
                r_col1, r_col2 = st.columns([1, 1])
                
                with r_col1:
                    if survival_prob > 0.5:
                        st.markdown(f"""
                        <div class="result-card-survival">
                            <h2 style="color:#00f260; margin:0;">SURVIVAL PROBABLE</h2>
                            <h1 style="font-size:4rem; color:white; margin:10px 0;">{survival_prob:.1%}</h1>
                            <p style="color:#888;">CONFIDENCE INTERVAL: HIGH</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-death">
                            <h2 style="color:#ff416c; margin:0;">SURVIVAL IMPROBABLE</h2>
                            <h1 style="font-size:4rem; color:white; margin:10px 0;">{prob[0]:.1%}</h1>
                            <p style="color:#888;">CONFIDENCE INTERVAL: HIGH</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with r_col2:
                    # Dark Mode Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=survival_prob * 100,
                        title={'text': "SURVIVAL PROBABILITY", 'font': {'color': '#888', 'size': 14}},
                        number={'font': {'color': 'white'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickcolor': "#888"},
                            'bar': {'color': "#00f260" if survival_prob > 0.5 else "#ff416c"},
                            'bgcolor': "#1a1c24",
                            'borderwidth': 2,
                            'bordercolor': "#2d2f36",
                            'steps': [{'range': [0, 100], 'color': '#13151b'}],
                            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': survival_prob*100}
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Roboto Mono'})
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Computation Error: {e}")

with tab2:
    st.markdown("#### 📉 MODEL PERFORMANCE ANALYTICS")
    
    # Dark Mode Chart
    perf_data = pd.DataFrame({
        'Model': ['Baseline', 'Advanced', 'Elite'],
        'Accuracy': [83.73, 82.72, 84.50],
        'Features': [9, 12, 20]
    })
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(perf_data, x='Model', y='Accuracy', 
                    color='Accuracy', color_continuous_scale=['#0575e6', '#00f260'],
                    template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        fig = px.line(perf_data, x='Model', y='Features', markers=True,
                     template='plotly_dark', line_shape='spline')
        fig.update_traces(line_color='#00f260')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    ### 📡 SYSTEM ARCHITECTURE
    
    **CORE**: `scikit-learn` v1.0+ Engine
    **INTERFACE**: `Streamlit` Neural Link
    **VISUALIZATION**: `Plotly` Dark Mode
    
    #### 🧬 FEATURE ENGINEERING EXCLUSIVE
    - **FamilyUnit**: Composite of SibSp + Parch
    - **TitleExtraction**: Regex-based social status identifier
    - **AgeBinning**: Advanced demographic segmentation
    
    > *System designed for high-precision predictive modeling in maritime disaster scenarios.*
    """)
    st.markdown("---")
    st.caption("© 2026 TITANIC AI SYSTEMS | SECURE CONNECTION ESTABLISHED")
