import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Configuration (MUST BE FIRST)
st.set_page_config(
    page_title="🚀 Cosmic Explorations: Space Mission Dashboard 🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("space_missions_dataset.csv")
    df["Launch Date"] = pd.to_datetime(df["Launch Date"], format="%d-%m-%Y")
    return df

df = load_data()

# Custom Theme
def apply_theme():
    st.markdown("""
    <style>
    /* Main app */
    .stApp {
        background-color: #F5F7FA;
        color: #2D3748;
    }
    
    /* Compact sidebar */
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2D3748 !important;
        border-bottom: 2px solid #4A5568;
        padding-bottom: 0.3rem;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        min-height: 120px;
        margin-bottom: 1rem;
    }
    
    /* Chart containers */
    .stPlotlyChart {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        background-color: #FFFFFF;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# Title
st.title("🌌 Space Mission Analytics Dashboard")
st.markdown("""
    <div style="color: #4A5568; margin-bottom: 2rem;">
    Interactive analysis of space mission parameters and performance metrics
    </div>
""", unsafe_allow_html=True)

# Sidebar Filters
with st.sidebar:
    st.header("🔧 Filters")
    
    # Date Range
    start_date = df["Launch Date"].min().date()
    end_date = df["Launch Date"].max().date()
    date_range = st.date_input(
        "Launch Date Range",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date
    )
    
    # Mission Type
    mission_types = st.multiselect(
        "Mission Type",
        options=df["Mission Type"].unique(),
        default=df["Mission Type"].unique()
    )
    
    # Dynamic Target Selection
    available_targets = df[df["Mission Type"].isin(mission_types)]["Target Type"].unique()
    target_types = st.multiselect(
        "Target Type",
        options=available_targets,
        default=available_targets
    )

# Data Filtering
filtered_df = df[
    (df["Launch Date"].dt.date >= date_range[0]) &
    (df["Launch Date"].dt.date <= date_range[1]) &
    (df["Mission Type"].isin(mission_types)) &
    (df["Target Type"].isin(target_types))
]

# Key Metrics
st.subheader("🚀 Mission Overview")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Total Missions", 
        value=f"{filtered_df.shape[0]:,}",
        help="Total number of missions in selected filters"
    )

with col2:
    total_cost = filtered_df['Mission Cost (billion USD)'].sum()
    st.metric(
        label="Total Cost", 
        value=f"${total_cost:,.2f}B",
        help="Total budget of all missions in billion USD"
    )

col3, col4 = st.columns(2)
with col3:
    success_rate = filtered_df['Mission Success (%)'].mean()
    st.metric(
        label="Avg Success Rate", 
        value=f"{success_rate:.1f}%",
        help="Average success percentage across missions"
    )

with col4:
    avg_duration = filtered_df['Mission Duration (years)'].mean()
    st.metric(
        label="Avg Duration", 
        value=f"{avg_duration:.1f} yrs",
        help="Average mission duration in years"
    )

# Main Visualizations
st.subheader("📈 Mission Analysis")

# Visualization code remains the same as previous version
# [Include all your existing visualization code here]

# Machine Learning Section
st.header("🤖 Predictive Analytics")
ml_tab1, ml_tab2 = st.tabs(["Success Prediction", "Mission Clustering"])

with ml_tab1:
    st.subheader("Mission Success Predictor")
    
    # Prepare ML data
    df_ml = df.copy()
    df_ml['Success_Class'] = np.where(df_ml['Mission Success (%)'] >= 90, 1, 0)
    
    # Features and target
    features = ['Mission Cost (billion USD)', 'Distance from Earth (light-years)',
               'Mission Duration (years)', 'Crew Size', 'Payload Weight (tons)']
    X = df_ml[features]
    y = df_ml['Success_Class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.1%}")
        
    with col2:
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
    
    # Prediction interface
    st.subheader("Live Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            cost = st.number_input("Mission Cost (Billion USD)", min_value=0.0, value=500.0)
            distance = st.number_input("Distance (Light Years)", min_value=0.0, value=10.0)
        with col2:
            duration = st.number_input("Duration (Years)", min_value=0.0, value=5.0)
            crew = st.number_input("Crew Size", min_value=0, value=20)
        
        payload = st.number_input("Payload Weight (Tons)", min_value=0.0, value=50.0)
        
        if st.form_submit_button("Predict Success"):
            input_data = [[cost, distance, duration, crew, payload]]
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            if prediction == 1:
                st.success(f"High Success Probability ({probability:.0%})")
                st.write("Key factors contributing to success:")
                importances = model.feature_importances_
                for feature, importance in zip(features, importances):
                    st.write(f"- {feature}: {importance:.1%}")
            else:
                st.error(f"Potential Risk of Lower Success ({probability:.0%})")
                st.write("Key factors affecting risk:")
                importances = model.feature_importances_
                for feature, importance in zip(features, importances):
                    st.write(f"- {feature}: {importance:.1%}")

with ml_tab2:
    st.subheader("Mission Clustering Analysis")
    st.write("""
    ### Mission Similarity Analysis
    Groups missions with similar characteristics using machine learning
    """)
    
    # Example clustering visualization
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    cluster_features = ['Mission Cost (billion USD)', 'Distance from Earth (light-years)',
                       'Mission Duration (years)', 'Crew Size']
    X_cluster = df[cluster_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Cluster
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Visualize
    fig = px.scatter_3d(
        x=X_cluster.iloc[:, 0],
        y=X_cluster.iloc[:, 1],
        z=X_cluster.iloc[:, 2],
        color=clusters,
        labels={'x': cluster_features[0],
               'y': cluster_features[1],
               'z': cluster_features[2]},
        title="Mission Clusters in 3D Space"
    )
    st.plotly_chart(fig, use_container_width=True)

# Data Table (Moved after ML section)
st.subheader("📋 Mission Details")
st.dataframe(
    filtered_df,
    column_order=["Mission Name", "Launch Date", "Target Type", 
                 "Mission Type", "Mission Cost (billion USD)"],
    height=300,
    use_container_width=True
)
