"""
LULC Classification Web Application
FIXED VERSION - Correct GEE Authentication for Web Deployment
Author: AI Assistant  
Date: 2025-10-18
"""

import streamlit as st
import pandas as pd
import numpy as np
import ee
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

APP_TITLE = "🌍 Advanced LULC Classification System"
APP_VERSION = "1.0.1"  # Updated version

AVAILABLE_MODELS = [
    'RandomForest', 'SVM', 'XGBoost', 'GBM', 'CART', 
    'ANN', 'DNN', 'LightGBM', 'KNN', 'NaiveBayes',
    'LogisticRegression', 'AdaBoost', 'ExtraTrees', 'CatBoost'
]

SUPPORTED_SENSORS = {
    'Landsat 4-5 TM': {'blue': 1, 'green': 2, 'red': 3, 'nir': 4, 'swir1': 5, 'swir2': 7},
    'Landsat 8 OLI': {'blue': 2, 'green': 3, 'red': 4, 'nir': 5, 'swir1': 6, 'swir2': 7},
    'Landsat 9 OLI-2': {'blue': 2, 'green': 3, 'red': 4, 'nir': 5, 'swir1': 6, 'swir2': 7}
}

SPECTRAL_INDICES = ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'EVI', 'SAVI', 'BSI', 'NDSI', 'NBR', 'NDMI']

GLOBAL_LULC_SOURCES = {
    'Dynamic World': 'GOOGLE/DYNAMICWORLD/V1',
    'ESA WorldCover': 'ESA/WorldCover/v100',
    'MODIS Land Cover': 'MODIS/006/MCD12Q1'
}

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .auth-instructions {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'gee_auth': False,
        'gee_project': None,
        'data_loaded': False,
        'indices_ready': False,
        'training_ready': False,
        'models_trained': False,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'trained_models': {},
        'results': {},
        'feature_names': [],
        'bands_data': None,
        'X_data': None,
        'n_classes': 0,
        'predictions': None,
        'class_names': [],
        'auth_step': 'initial'
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ==================== GEE AUTHENTICATION (FIXED) ====================

def authenticate_gee():
    """
    Authenticate Google Earth Engine using notebook mode
    This is the correct method for web-based Streamlit apps
    """
    st.markdown("### 🔐 Google Earth Engine Authentication")
    
    st.markdown("""
    <div class="auth-instructions">
    <h4>📋 Authentication Steps:</h4>
    <ol>
        <li><strong>Click the "Start Authentication" button below</strong></li>
        <li><strong>Click the generated link</strong> (opens in new tab)</li>
        <li><strong>Sign in</strong> with your Google account</li>
        <li><strong>Select or create a Cloud Project</strong> for authentication</li>
        <li><strong>Click "Generate Token"</strong></li>
        <li><strong>Grant permissions</strong> when asked</li>
        <li><strong>Copy the authorization code</strong></li>
        <li><strong>Paste it back here</strong> and press Enter</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Start Authentication", type="primary", use_container_width=True):
            st.session_state.auth_step = 'authenticating'
            
            try:
                # Use notebook mode for web-based authentication
                st.info("🔄 Initiating authentication flow...")
                
                # This will generate a URL for the user to visit
                ee.Authenticate(auth_mode='notebook', quiet=False)
                
                st.session_state.auth_step = 'code_entry'
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Authentication initiation failed: {str(e)}")
                st.session_state.auth_step = 'initial'
    
    with col2:
        st.info("💡 **Tip:** You need a Google Earth Engine account (free at [earthengine.google.com](https://earthengine.google.com))")
    
    # If authentication was started, show code entry
    if st.session_state.auth_step == 'code_entry':
        st.markdown("---")
        st.markdown("### 🔑 Enter Authorization Code")
        
        auth_code = st.text_input(
            "Paste the authorization code here:",
            type="password",
            help="Copy the code from the authentication page and paste it here"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Verify Code", type="primary", use_container_width=True):
                if auth_code:
                    try:
                        # Try to initialize with a project
                        project_id = st.text_input(
                            "Enter your GEE Project ID:",
                            help="This is your Google Cloud Project ID (e.g., 'ee-myusername')"
                        )
                        
                        if project_id:
                            ee.Initialize(project=project_id)
                            st.session_state.gee_auth = True
                            st.session_state.gee_project = project_id
                            st.session_state.auth_step = 'complete'
                            st.success("✅ Successfully authenticated!")
                            st.balloons()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {str(e)}")
                else:
                    st.warning("⚠️ Please enter the authorization code")
        
        with col2:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.auth_step = 'initial'
                st.rerun()

def show_gee_status():
    """Display GEE authentication status in sidebar"""
    st.header("🔐 GEE Authentication")
    
    if not st.session_state.gee_auth:
        st.markdown('<div class="warning-box">⚠️ Not Authenticated</div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ How to Authenticate", expanded=False):
            st.markdown("""
            **For web-based Streamlit apps:**
            
            1. Go to "Home" page
            2. Click "Authenticate GEE" button
            3. Follow the authentication flow
            4. Enter the authorization code
            
            **Note:** This uses 'notebook' mode authentication, 
            which is designed for web environments.
            """)
    else:
        st.markdown('<div class="success-box">✅ Authenticated</div>', unsafe_allow_html=True)
        if st.session_state.gee_project:
            st.caption(f"📁 Project: {st.session_state.gee_project}")
        
        try:
            ee.Initialize(project=st.session_state.gee_project)
        except:
            st.warning("⚠️ Session expired. Please re-authenticate.")
            st.session_state.gee_auth = False
            st.rerun()

# ==================== HELPER FUNCTIONS ====================

def calculate_indices(bands_dict):
    """Calculate all spectral indices from band data"""
    indices = {}
    eps = 1e-10
    
    b = bands_dict.get('blue', 0)
    g = bands_dict.get('green', 0)
    r = bands_dict.get('red', 0)
    nir = bands_dict.get('nir', 0)
    s1 = bands_dict.get('swir1', 0)
    s2 = bands_dict.get('swir2', 0)
    
    indices['NDVI'] = (nir - r) / (nir + r + eps)
    indices['NDWI'] = (g - nir) / (g + nir + eps)
    indices['MNDWI'] = (g - s1) / (g + s1 + eps)
    indices['NDBI'] = (s1 - nir) / (s1 + nir + eps)
    indices['EVI'] = 2.5 * ((nir - r) / (nir + 6*r - 7.5*b + 1 + eps))
    indices['SAVI'] = 1.5 * ((nir - r) / (nir + r + 0.5 + eps))
    indices['BSI'] = ((s1 + r) - (nir + b)) / ((s1 + r) + (nir + b) + eps)
    indices['NDSI'] = (g - s1) / (g + s1 + eps)
    indices['NBR'] = (nir - s2) / (nir + s2 + eps)
    indices['NDMI'] = (nir - s1) / (nir + s1 + eps)
    
    return indices

def get_model(model_name):
    """Get machine learning model by name"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GBM': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'CART': DecisionTreeClassifier(random_state=42),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500),
        'DNN': MLPClassifier(hidden_layer_sizes=(256,128,64), random_state=42, max_iter=1000),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'NaiveBayes': GaussianNB(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'CatBoost': cb.CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    }
    return models.get(model_name)

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive accuracy metrics"""
    metrics = {
        'Overall Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 2:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['ROC-AUC'] = 0.0
    
    metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig

def plot_metrics_comparison(results_dict):
    """Plot model performance comparison heatmap"""
    metrics_names = ['Overall Accuracy', 'Precision', 'Recall', 'F1-Score', 
                     'Kappa', 'MCC', 'Balanced Accuracy']
    
    data = []
    model_names = []
    
    for model_name, metrics in results_dict.items():
        model_names.append(model_name)
        row = [metrics.get(m, 0) for m in metrics_names]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(model_names)*0.6)))
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn',
                xticklabels=[m.replace(' ', '\n') for m in metrics_names],
                yticklabels=model_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Multi-Model Land Use Land Cover Classification Platform v{APP_VERSION}</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://earthengine.google.com/static/images/earth-engine-logo.png", width=180)
        st.title("🧭 Navigation")
        
        # GEE Authentication Status
        show_gee_status()
        
        st.markdown("---")
        
        # Page Navigation
        page = st.radio(
            "📑 Select Page",
            [
                "🏠 Home & Authentication",
                "📁 Data Upload",
                "🧮 Calculate Indices",
                "📊 Training Data",
                "🤖 Train Models",
                "📈 Results & Evaluation",
                "🔍 Classify & Predict",
                "💾 Export Results"
            ],
            key="nav_radio"
        )
        
        st.markdown("---")
        
        # System Status
        st.header("📊 System Status")
        status_items = {
            "GEE Auth": st.session_state.gee_auth,
            "Data Loaded": st.session_state.data_loaded,
            "Indices Ready": st.session_state.indices_ready,
            "Training Ready": st.session_state.training_ready,
            "Models Trained": st.session_state.models_trained
        }
        
        for item, status in status_items.items():
            icon = "✅" if status else "❌"
            color = "#28a745" if status else "#dc3545"
            st.markdown(f'<p style="color:{color};">{icon} {item}</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption(f"Version {APP_VERSION} | Powered by Streamlit & GEE")
    
    # Main Content Router
    if "Home" in page or "Authentication" in page:
        show_home()
    elif "Data Upload" in page:
        show_data_upload()
    elif "Calculate Indices" in page:
        show_indices()
    elif "Training Data" in page:
        show_training_data()
    elif "Train Models" in page:
        show_model_training()
    elif "Results" in page:
        show_results()
    elif "Classify" in page:
        show_prediction()
    elif "Export" in page:
        show_export()

# ==================== PAGE FUNCTIONS ====================

def show_home():
    """Home page with authentication and features"""
    
    st.header("Welcome to LULC Classification System")
    
    # Authentication Section
    if not st.session_state.gee_auth:
        authenticate_gee()
        st.markdown("---")
    
    # Features Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✨ Key Features")
        st.markdown("""
        #### 🤖 15+ Machine Learning Models
        - Random Forest, XGBoost, SVM, GBM
        - Neural Networks (ANN, DNN)
        - LightGBM, CatBoost, AdaBoost
        - KNN, Naive Bayes, and more
        
        #### 🌈 10+ Spectral Indices
        - NDVI, NDWI, NDBI, EVI, SAVI
        - MNDWI, BSI, NDSI, NBR, NDMI
        
        #### 📊 Comprehensive Evaluation
        - 10+ accuracy metrics
        - Confusion matrix heatmaps
        - Feature importance analysis
        - Model comparison visualizations
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        #### Workflow Steps:
        
        **1️⃣ Authenticate** (above if not done)
        
        **2️⃣ Upload Data**
        - Navigate to "Data Upload"
        - Upload Landsat imagery
        
        **3️⃣ Calculate Indices**
        - Auto-calculate spectral indices
        
        **4️⃣ Training Data**
        - Generate sample data or upload
        
        **5️⃣ Train Models**
        - Select and train multiple models
        
        **6️⃣ Evaluate & Export**
        - Compare results and export
        """)
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card"><h2>15+</h2><p>ML Models</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>10+</h2><p>Indices</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>10</h2><p>Metrics</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>3</h2><p>Sensors</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card"><h2>∞</h2><p>Coverage</p></div>', unsafe_allow_html=True)

def show_data_upload():
    st.header("📁 Data Upload")
    if not st.session_state.gee_auth:
        st.warning("⚠️ Please authenticate GEE first from the Home page")
        return
    st.info("Data upload functionality - Implementation continues...")

def show_indices():
    st.header("🧮 Calculate Indices")
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first")
        return
    st.info("Indices calculation - Implementation continues...")

def show_training_data():
    st.header("📊 Training Data")
    if not st.session_state.indices_ready:
        st.warning("⚠️ Please calculate indices first")
        return
    st.info("Training data preparation - Implementation continues...")

def show_model_training():
    st.header("🤖 Model Training")
    if not st.session_state.training_ready:
        st.warning("⚠️ Please prepare training data first")
        return
    st.info("Model training - Implementation continues...")

def show_results():
    st.header("📈 Results & Evaluation")
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
        return
    st.info("Results evaluation - Implementation continues...")

def show_prediction():
    st.header("🔍 Classify & Predict")
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
        return
    st.info("Classification - Implementation continues...")

def show_export():
    st.header("💾 Export Results")
    if not st.session_state.models_trained:
        st.warning("⚠️ No results to export")
        return
    st.info("Export functionality - Implementation continues...")

# ==================== RUN APP ====================

if __name__ == "__main__":
    main()
