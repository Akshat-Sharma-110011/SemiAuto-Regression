# SemiAuto-Regression

**SemiAuto-Regression** is a Python-based microservice that provides a semi-automated pipeline for regression modeling. It orchestrates the end-to-end ML lifecycle—from data ingestion and preprocessing through model training, evaluation, and report generation—while allowing user interaction at key decision points.

Key features include:

- **Regression-focused AutoML**: Automates training and evaluation of multiple regression models and generates a ranked leaderboard of results.
- **User-Guided Workflow**: Combines automation with manual control over model choices, hyperparameters, and feature selection.
- **FastAPI Integration**: RESTful endpoints for triggering pipeline stages and retrieving results.
- **HTML-Based UI**: Visual dashboard served via Jinja2 templates for progress tracking and report downloads.
- **Modular Architecture**: Clean separation of components like data handling, model training, evaluation, and reporting.
- **Docker-Compatible**: Containerized setup for reproducible deployment.

---

## 🧠 Architecture Overview

```
SemiAuto-Regression/
├── app.py                # FastAPI app launcher
├── config/               # YAML/JSON configuration files
├── data/                 # Raw, interim, cleaned, and processed data folders
├── intel.yaml            # Master config for paths and control flags
├── model/                # Trained model artifacts, encoders, transformers
├── reports/              # Auto-generated evaluation and summary reports
├── requirements.txt      # Python dependencies
├── semiauto_regression/  # Core ML pipeline modules
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   ├── model_optimization.py
│   └── projectflow_reports.py
├── static/               # CSS/JS assets for web interface
├── templates/            # Jinja2 templates for HTML rendering
├── tests/                # Unit tests
└── Dockerfile            # Container build file
```

---

## ⚙️ Installation

### ✅ Requirements
- Python 3.10+
- pip / virtualenv
- Docker (optional)
- OS support for libraries like CatBoost (e.g., `libgomp1`)

### 📥 Install with pip

```bash
git clone https://github.com/Akshat-Sharma-110011/SemiAuto-Regression.git
cd SemiAuto-Regression
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🐳 Run via Docker

```bash
docker build -t semiauto-regression .
docker run --rm -p 8030:8030 semiauto-regression
```

---

## 🚀 Usage

### 🌐 Web App
Launch FastAPI and access:
```
http://localhost:8030/
```

### 📡 API Endpoints
- `POST /train` – Start training pipeline
- `GET /status` – Get training status
- `GET /predict` – Perform inference
- `GET /report` – Download performance reports

### 🔁 CLI Example

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8030
```

```bash
curl -X POST http://localhost:8030/train \
     -H "Content-Type: application/json" \
     -d '{"dataset": "data_boston", "model": "catboost", "epochs": 50}'
```

---

## 🛠️ Customization

- Update `intel.yaml` to configure dataset, output paths, models
- Add your own dataset to `data/raw/data_<name>/original.csv`
- Extend YAML in `references/feature_store_<dataset>.yaml` for metadata
- Plug new models by editing `model_building.py` and `model_evaluation.py`
- Modify UI templates under `templates/` for branding/custom views

---

## ⚠️ Limitations & Roadmap

- 🧪 Currently supports only **regression** tasks
- 🤖 Limited AutoML algorithms (CatBoost, etc.); no full ensembling yet
- 🧹 Feature engineering is basic — you’ll need to preprocess data
- 📉 Does not yet support classification or large-scale distributed ML
- 📦 Future plans:
  - Classification support
  - Advanced hyperparameter optimization
  - CI/CD integration
  - Airflow or DVC compatibility
  - Cloud-native pipeline exports

---

## 📚 Credits & References

This project is inspired by open AutoML frameworks such as:

- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) – for leaderboard-based model selection
- [AutoML: A Survey of the State-of-the-Art](https://arxiv.org/abs/1904.12054) – K. Hutter et al., 2019

> “Automated Machine Learning (AutoML) aims to automate the end-to-end process of applying machine learning to real-world problems.” – Hutter et al.

---

## 🪪 License

This project is licensed under the MIT License.
