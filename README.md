# 🌍 Advanced LULC Classification System

A comprehensive web-based Land Use Land Cover (LULC) classification platform powered by machine learning and Google Earth Engine.

## 🚀 Features

- **15+ Machine Learning Models**: Random Forest, SVM, XGBoost, GBM, Neural Networks, and more
- **10+ Spectral Indices**: Automated calculation of NDVI, NDWI, NDBI, EVI, SAVI, etc.
- **Multi-Sensor Support**: Landsat 4-5 TM, Landsat 8 OLI, Landsat 9 OLI-2
- **Comprehensive Evaluation**: 10+ accuracy metrics with visualizations
- **Google Earth Engine Integration**: Direct access to satellite imagery and global LULC datasets
- **Advanced Analytics**: Feature importance, sensitivity analysis, model comparison
- **User-Friendly Interface**: Zero installation required - runs in web browser

## 📋 Prerequisites

- Python 3.8 or higher
- Google Earth Engine account (free at https://earthengine.google.com/)
- GitHub account (for deployment)

## 🛠️ Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/lulc-classifier-web.git
cd lulc-classifier-web

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine
earthengine authenticate

# Run application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ☁️ Cloud Deployment (Streamlit Community Cloud)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/lulc-classifier-web.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/lulc-classifier-web`
5. Main file path: `app.py`
6. Click "Deploy"

Your app will be live at: `https://YOUR_USERNAME-lulc-classifier-web.streamlit.app`

## 📖 Usage Guide

### 1. Authenticate Google Earth Engine
- Click "Authenticate GEE" in the sidebar
- Sign in with your Google account
- Grant necessary permissions

### 2. Upload Data
- Navigate to "Data Upload" page
- Upload Landsat GeoTIFF imagery
- Or fetch directly from Google Earth Engine

### 3. Calculate Indices
- System automatically calculates 10+ spectral indices
- Select which indices to include in your model

### 4. Prepare Training Data
- Upload shapefile with training polygons
- Or generate sample data for testing

### 5. Train Models
- Select multiple models to train
- Enable hyperparameter tuning (optional)
- Monitor training progress

### 6. Evaluate Results
- Compare model performance
- View confusion matrices
- Analyze feature importance

### 7. Classify & Export
- Generate classification map
- Export results in multiple formats

## 🎯 Supported Models

### Tree-Based
- Random Forest
- CART (Decision Tree)
- Extra Trees

### Boosting
- XGBoost
- Gradient Boosting Machine (GBM)
- LightGBM
- CatBoost
- AdaBoost

### Neural Networks
- Artificial Neural Network (ANN)
- Deep Neural Network (DNN)

### Classical ML
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression

## 📊 Accuracy Metrics

- Overall Accuracy
- Precision (User's Accuracy)
- Recall (Producer's Accuracy)
- F1-Score
- Kappa Coefficient
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy
- ROC-AUC Score
- Confusion Matrix

## 🌈 Spectral Indices

| Index | Purpose |
|-------|---------|
| NDVI | Vegetation health |
| NDWI | Water content |
| MNDWI | Water bodies |
| NDBI | Built-up areas |
| EVI | Enhanced vegetation |
| SAVI | Soil-adjusted vegetation |
| BSI | Bare soil |
| NDSI | Snow cover |
| NBR | Burn severity |
| NDMI | Moisture content |

## 🔧 Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Upload size limits
- Server settings

## 📁 Project Structure

```
lulc-classifier-web/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Google Earth Engine for satellite imagery
- Streamlit for the web framework
- scikit-learn, XGBoost, LightGBM for ML algorithms

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🌟 Star this repository if you find it helpful!

---

**Built with ❤️ using Streamlit and Google Earth Engine**
