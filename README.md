# Insurance Charge Predictor — End to End ML Deployment

A production-deployed ML system that predicts medical insurance charges
via a REST API. Built with a tuned sklearn Pipeline, served via FastAPI,
and deployed on Render.

---

## Live API

```
POST https://insurance-charge-predictor-api.onrender.com/predict
```

**Sample Request:**
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

**Sample Response:**
```json
{
    "predicted_charges": 17809.85
}
```

---

## C1 — Hyperparameter Tuning with GridSearchCV

Instead of relying on sklearn's default parameters, GridSearchCV
systematically searched 27 combinations across 5 folds — 135 total
training runs — to find the optimal configuration.

### Parameter Grid

| Parameter | Values Tried | Count |
|---|---|---|
| `n_estimators` | 100, 200, 300 | 3 |
| `max_depth` | None, 5, 10 | 3 |
| `min_samples_split` | 2, 5, 10 | 3 |

**3 × 3 × 3 = 27 combinations × 5 folds = 135 total training runs**

### Results

| Model | R² | How Measured |
|---|---|---|
| Untuned Random Forest | 0.8382 | 5-fold CV |
| **Tuned Random Forest** | **0.8458** | 5-fold CV (GridSearchCV) |

### Best Parameters Found

```
n_estimators      = 300
max_depth         = 5
min_samples_split = 5
```

### Code

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators'     : [100, 200, 300],
    'model__max_depth'        : [None, 5, 10],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best R²        :", round(grid_search.best_score_, 4))
```

---

## C2 — Model Serialisation

The best pipeline — preprocessor + tuned RandomForest — was frozen
to disk using joblib. This allows the model to be loaded anywhere:
locally, on a server, or inside a Docker container.

```python
import joblib

joblib.dump(grid_search.best_estimator_, 'model.pkl')
print("Model saved to model.pkl")
```

**What gets saved inside model.pkl:**
- ColumnTransformer (StandardScaler for numeric, OneHotEncoder for categorical)
- Tuned RandomForestRegressor (best params from GridSearchCV)

> **Why joblib over pickle?**
> joblib is optimised for large numpy arrays — faster and more efficient
> than Python's built-in pickle for sklearn models.

---

## C3 — FastAPI Endpoint

A REST API was built using FastAPI. It accepts patient details,
runs them through the full pipeline, and returns a predicted charge.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('model.pkl')
app = FastAPI()

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post('/predict')
def predict(data: InsuranceInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {'predicted_charges': round(prediction, 2)}
```

### What Each Part Does

| Component | Purpose |
|---|---|
| `BaseModel` | Validates and defines the shape of incoming data |
| `joblib.load` | Loads the full tuned pipeline from disk |
| `@app.post('/predict')` | Creates a POST endpoint at /predict |
| `data.dict()` | Converts input to dictionary for DataFrame |
| `model.predict` | Runs the full pipeline — preprocess + predict |

---

## C4 — Deployment on Render

The API was deployed to Render — a cloud platform that hosts the
service 24/7 with automatic deploys on every GitHub push.

### Deployment Configuration

| Setting | Value |
|---|---|
| **Environment** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port 10000` |

### requirements.txt

```
fastapi
uvicorn
pydantic
pandas
scikit-learn==1.6.1
```

> **Why pin scikit-learn?**
> A .pkl file is tightly coupled to the sklearn version that created it.
> Pinning ensures training and serving environments are identical —
> preventing silent failures in production.

---

## C5 — Show-off Checkpoint

A live prediction request from Python:

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

response = requests.post(
    "https://insurance-charge-predictor-api.onrender.com/predict",
    json=payload
)

print(response.json())
# {'predicted_charges': 17809.85}
```

---

## Portfolio Statement

> *"Built and deployed an end-to-end insurance charge prediction API.
> Tuned a Random Forest with GridSearchCV across 135 combinations.
> Cross-validated R² improved from 0.8382 to 0.8458. Deployed as a
> REST API on Render — live and callable from anywhere."*

---

## End to End Architecture

```
Raw CSV data
    ↓
EDA + Feature Analysis
    ↓
ColumnTransformer Pipeline (systematic preprocessing)
    ↓
GridSearchCV (135 runs, best params selected with evidence)
    ↓
Tuned RandomForest (R² = 0.8458, cross-validated)
    ↓
Serialised to model.pkl
    ↓
FastAPI endpoint (/predict)
    ↓
Deployed on Render
    ↓
Live URL — callable from anywhere
```
