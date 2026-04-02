# Insurance Charge Predictor API

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live API](https://img.shields.io/badge/Live%20API-Online-brightgreen?style=flat)](https://insurance-charge-predictor-api.onrender.com/predict)

> **Production-deployed end-to-end ML system** that predicts medical insurance charges from patient demographics via a REST API. Trained with a tuned sklearn pipeline (GridSearchCV, R² = 0.8458), served via FastAPI, and deployed on Render.

---

## 🚀 Live Demo

**API Endpoint:** `POST https://insurance-charge-predictor-api.onrender.com/predict`

Try it instantly from your terminal:

```bash
curl -X POST https://insurance-charge-predictor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 19, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest"}'
```

**Response:**
```json
{"predicted_charges": 17809.85}
```

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Key Engineering Decisions](#key-engineering-decisions)

---

## Overview

This project demonstrates a **complete ML engineering workflow** from raw data to a live, callable API:

1. **Data preprocessing** using a `ColumnTransformer` pipeline (StandardScaler + OneHotEncoder)
2. **Hyperparameter tuning** via GridSearchCV across 135 combinations (27 configs × 5-fold CV)
3. **Model serialisation** with joblib for reproducible, environment-safe deployment
4. **REST API** built with FastAPI + Pydantic for type-safe request validation
5. **Cloud deployment** on Render with pinned dependency versions to prevent environment drift

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Training Pipeline                    │
│                                                          │
│  Raw CSV  ──►  EDA  ──►  ColumnTransformer Pipeline     │
│                              │                           │
│                    ┌─────────▼──────────┐                │
│                    │   GridSearchCV      │                │
│                    │  27 configs × 5 CV  │                │
│                    │  = 135 total runs   │                │
│                    └─────────┬──────────┘                │
│                              │                           │
│                    Best Params Selected                  │
│                    R² = 0.8458 (5-fold CV)               │
│                              │                           │
│                    joblib.dump() ──► model.pkl           │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                    Serving Pipeline                      │
│                                                          │
│  HTTP POST /predict                                      │
│       │                                                  │
│       ▼                                                  │
│  Pydantic Validation  ──►  joblib.load(model.pkl)        │
│                                  │                       │
│                         Preprocess + Predict             │
│                                  │                       │
│                         JSON Response                    │
└─────────────────────────────────────────────────────────┘
                              │
                    Deployed on Render
                    Auto-deploy on git push
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Framework | scikit-learn 1.6.1 | Pipeline, preprocessing, RandomForest |
| Hyperparameter Tuning | GridSearchCV | 135-run systematic search |
| API Framework | FastAPI | REST endpoint with auto-generated docs |
| Data Validation | Pydantic | Type-safe request schema |
| Model Serialisation | joblib | Persist full pipeline to disk |
| Data Manipulation | pandas | DataFrame construction for inference |
| Deployment | Render | Cloud hosting with auto-deploy |
| Notebook | Jupyter / Google Colab | Training and experimentation |

---

## Model Performance

### Hyperparameter Search Space

| Parameter | Values Tried | Count |
|---|---|---|
| `n_estimators` | 100, 200, 300 | 3 |
| `max_depth` | None, 5, 10 | 3 |
| `min_samples_split` | 2, 5, 10 | 3 |

**Total training runs:** 3 × 3 × 3 = 27 combinations × 5 folds = **135 runs**

### Results

| Model | R² Score | Evaluation Method |
|---|---|---|
| Baseline Random Forest (default params) | 0.8382 | 5-fold cross-validation |
| **Tuned Random Forest (GridSearchCV)** | **0.8458** | 5-fold cross-validation |

### Best Parameters

```
n_estimators      = 300
max_depth         = 5
min_samples_split = 5
```

---

## API Reference

### `POST /predict`

Predicts insurance charges based on patient demographics.

**Request Body**

```json
{
  "age": 19,
  "sex": "female",
  "bmi": 27.9,
  "children": 0,
  "smoker": "yes",
  "region": "southwest"
}
```

**Field Descriptions**

| Field | Type | Description | Example Values |
|---|---|---|---|
| `age` | int | Age of the primary beneficiary | 18–64 |
| `sex` | str | Biological sex | `"male"`, `"female"` |
| `bmi` | float | Body mass index | 15.0–55.0 |
| `children` | int | Number of dependents | 0–5 |
| `smoker` | str | Smoking status | `"yes"`, `"no"` |
| `region` | str | US region of coverage | `"northeast"`, `"northwest"`, `"southeast"`, `"southwest"` |

**Response**

```json
{
  "predicted_charges": 17809.85
}
```

**Interactive Docs:** Available at `/docs` (Swagger UI) when running locally.

---

## Project Structure

```
insurance-charge-predictor-api/
│
├── 01_train_tune_export_model.ipynb  # Full training pipeline: EDA → tuning → serialisation
├── main.py                           # FastAPI application with /predict endpoint
├── model.pkl                         # Serialised sklearn pipeline (preprocessor + model)
├── requirements.txt                  # Pinned production dependencies
├── .gitignore                        # Standard Python ignores
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## Local Setup

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/SathyaPrakashD/insurance-charge-predictor-api.git
cd insurance-charge-predictor-api

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API server
uvicorn main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

**Test with Python:**

```python
import requests

payload = {
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())  # {'predicted_charges': 17809.85}
```

---

## Key Engineering Decisions

**Why GridSearchCV over manual tuning?**
Systematic search eliminates guesswork and provides evidence for parameter choices. 135 training runs give confidence that the selected configuration is objectively optimal within the defined search space.

**Why pin scikit-learn to 1.6.1?**
`.pkl` files are tightly coupled to the sklearn version that created them. A version mismatch between training and serving environments causes silent failures. Pinning ensures reproducibility across any deployment target.

**Why joblib over pickle?**
joblib is optimised for large NumPy arrays — significantly faster and more memory-efficient than Python’s built-in `pickle` for sklearn pipelines that contain fitted transformers and estimators.

**Why a full sklearn Pipeline?**
Bundling preprocessing (StandardScaler, OneHotEncoder) with the model in a single `Pipeline` object ensures the same transformations are applied consistently during both training and inference, eliminating train/serve skew.

**Why FastAPI?**
FastAPI provides automatic request validation via Pydantic, auto-generated OpenAPI docs (`/docs`), and async support — making it the modern standard for Python ML APIs.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built as part of an end-to-end ML engineering portfolio — demonstrating the full journey from raw data to a live, production-ready API.*
