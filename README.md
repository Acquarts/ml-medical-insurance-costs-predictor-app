# 🏥 Medical Insurance Cost Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run-4285F4?logo=googlecloud)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker)

Web application to predict medical insurance costs using Machine Learning, deployed on Google Cloud Run.

🔗 **Live Demo:** [insurance-predictor-562289298058.us-central1.run.app](https://insurance-predictor-562289298058.us-central1.run.app/)

## ✨ Features

- ML-based medical insurance cost prediction
- Interactive web interface with Streamlit
- Gradient Boosting model with 90% accuracy (R²)
- Deployed on Google Cloud Run

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML** | scikit-learn, XGBoost, pandas, numpy |
| **Web** | Streamlit |
| **Cloud** | Google Cloud Run, Cloud Build |
| **Containers** | Docker |

## 📁 Project Structure

```
├── app.py                 # Streamlit application
├── train.py               # Training script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container for Cloud Run
├── Dockerfile.training    # Container for training
├── .env                   # Environment variables (don't push to git)
├── data/
│   └── insurance.csv      # Dataset
└── model/
    ├── model.joblib       # Trained model
    └── feature_names.joblib
```

## 🚀 Local Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/ai-insurance-cost-predictor.git
cd ai-insurance-cost-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
# https://www.kaggle.com/datasets/mirichoi0218/insurance
# Save as data/insurance.csv

# 5. Train model (optional, already included)
python train.py --data-path=data/insurance.csv --model-dir=model

# 6. Run application
streamlit run app.py
```

App will be available at: http://localhost:8501

## ☁️ Deploy to Google Cloud Run

### Requirements

- Google Cloud account with billing enabled
- gcloud CLI installed and configured

### Steps

```powershell
# 1. Set project
gcloud config set project YOUR-PROJECT-ID

# 2. Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com storage.googleapis.com containerregistry.googleapis.com

# 3. Build image in the cloud
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/insurance-app .

# 4. Deploy to Cloud Run
gcloud run deploy insurance-predictor --image gcr.io/YOUR-PROJECT-ID/insurance-app --platform managed --region us-central1 --allow-unauthenticated --memory 1Gi --port 8080
```

## 📊 ML Model

### Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.90 |
| MAE | $2,530 |
| RMSE | $4,269 |

### Feature Importance

1. 🚬 Smoker (~70%)
2. ⚖️ BMI (~15%)
3. 📅 Age (~10%)
4. 📍 Other (~5%)

## 📋 Input Variables

| Variable | Type | Description |
|----------|------|-------------|
| age | int | Age (18-100) |
| sex | str | Sex (Male/Female) |
| bmi | float | Body Mass Index |
| children | int | Number of children (0-5) |
| smoker | str | Smoker (Yes/No) |
| region | str | Region (Northeast/Northwest/Southeast/Southwest) |

## 💰 Estimated GCP Costs

| Service | Approximate Cost |
|---------|------------------|
| Cloud Run | ~$0-5/month |
| Cloud Build | ~$0.003/build |
| Container Registry | ~$0.10/GB |

## 📂 Dataset

Medical Cost Personal Dataset from Kaggle:
https://www.kaggle.com/datasets/mirichoi0218/insurance

## 👤 Author

**Adrian Zambrana**

## 📄 License

MIT License
