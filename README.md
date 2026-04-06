# 🛒 E-Commerce Product Return Prediction (End-to-End ML Pipeline)

## 📌 Executive Summary
In the e-commerce industry, **reverse logistics (processing returns) is a massive profit killer**. Shipping an item back, inspecting it, repackaging it, and restocking it often costs more than the margin of the product itself. 

This project is a complete, end-to-end Machine Learning pipeline designed to predict the probability that a customer will return an item **at the exact moment they attempt to checkout**. By identifying high-risk transactions in real-time, businesses can trigger dynamic, cost-saving interventions:
* **Friction:** Revoking "Free Return Shipping" for serial returners.
* **Incentives:** Offering a targeted "Keep It" discount to users with a high return probability.
* **Marketing:** Suppressing ad-retargeting spend on users who have a negative lifetime value due to high return rates.

---

## 🌐 Live API Deployment

The machine learning model is actively deployed as a RESTful API using FastAPI and is hosted on Render.

* **API Endpoint:** `https://product-return-prediction.onrender.com`
* **Interactive API Docs (Swagger UI):** `https://product-return-prediction.onrender.com/docs`

*(Note: Free-tier Render instances spin down after 15 minutes of inactivity. The first request may take ~50 seconds to wake the server).*

---

## 📊 Model Performance & Business Impact

In highly imbalanced datasets (returns account for only ~2.3% of all orders), global accuracy is a misleading "trap." While this model achieves a **98% overall accuracy**, the true value lies in its performance on the minority class (Returns). 

By evaluating the optimal dynamic threshold, the model achieved the following business-critical metrics on unseen validation data:

* **Precision (64%):** When the model flags a transaction as a return risk, it is correct 64% of the time. This high precision minimizes "False Positives" (576), meaning the business avoids applying unnecessary friction to perfectly good customers.
* **Recall (40%):** The model successfully identifies 40% of all actual returns before they happen.
* **F1-Score (0.49):** A highly stable harmonic mean for an imbalanced e-commerce environment.

### The Confusion Matrix translated to Business Value:
Out of 109,905 historical checkout events evaluated:
* **True Positives (1,009):** Returns successfully caught *before* checkout. These represent direct financial savings through intervention.
* **True Negatives (106,808):** Normal purchases successfully identified. No friction added.
* **False Positives (576):** False alarms. A low number, representing an acceptable margin of risk for targeted interventions.
* **False Negatives (1,512):** Returns that slipped through the cracks. While standard reverse-logistics costs apply here, the 1,009 True Positives represent a massive upgrade over having no predictive system at all.

---

## 🧠 Key Engineering Innovations

Building a model for time-series e-commerce data requires navigating severe data leakage and class imbalance. Here is how this pipeline solves those core ML engineering challenges:

### 1. Eliminating Target Leakage (The "Time-Travel" Fix)
Raw e-commerce data logs returns as completely separate, future transactions (invoices starting with 'C'). Feeding this directly into a model causes severe data leakage, as the model learns to look for a 'C' invoice that doesn't exist at checkout time. 
* **The Architecture:** Engineered a custom data transformation module that isolates future cancellation invoices, extracts the exact Customer-Product pairs, merges the target `1` label back onto the *historical* purchase rows, and drops the future rows entirely. This perfectly simulates real-time, point-in-time inference.

### 2. Temporal Splitting & Imbalance Handling
Time-series purchase data cannot be evaluated using a random `train_test_split`, as it allows the model to look into the future to predict the past. 
* **The Architecture:** Implemented a strict **Temporal Split** (sorting by date and masking the last 20% of the timeline) to maintain chronological integrity. To handle the extreme 1.68% training class imbalance, the pipeline utilizes **ADASYN** (Adaptive Synthetic Sampling) combined with a **RandomUnderSampler** to build clear decision boundaries without overlapping noise.

### 3. Stateful Feature Store for Real-Time Inference
Models frequently crash in production when the API encounters a new customer, a new product, or an out-of-bounds numerical input that the scaler hasn't seen before.
* **The Architecture:** Built a **Stateful Feature Store** (`feature_store.pkl`). During training, historical business metrics (e.g., a customer's average order value, historical return rate, dynamic bins) are calculated and pickled. During real-time inference, the FastAPI endpoint queries this store to map historical data to the incoming JSON payload. It gracefully imputes global fallback averages (`.fillna()`) and safely clips bin boundaries (`.clip()`) to guarantee zero downtime.

---

## 🏗️ Project Architecture & ML Lifecycle

The repository is modularized using modern MLOps principles, separating concerns across data ingestion, transformation, training, and deployment:

```text
product-return-prediction/
│
├── artifacts/                          # Serialized pipeline outputs 
│   ├── preprocessor.pkl                # Pickled ColumnTransformer (RobustScaling/OrdinalEncoding)
│   ├── feature_store.pkl               # Pickled Stateful Feature Store (Business logic maps)
│   └── model.pkl                       # Pickled LightGBM Classifier
│
├── logs/                               # Auto-generated runtime logs with exact tracebacks
│
├── notebook/                           
│   └── product_return_prediction.ipynb # Exploratory Data Analysis & evaluation metrics
│
├── src/                                # Core source code module
│   ├── exception.py                    # Custom exception handling capturing line numbers
│   ├── logger.py                       # Timestamped logging configuration
│   ├── utils.py                        # Object serialization (Dill) and global constants
│   │
│   ├── components/                     
│   │   ├── data_ingestion.py           # Ingests raw data, applies temporal splits
│   │   ├── data_transformation.py      # Fixes leakage, builds feature store, applies ADASYN
│   │   └── model_trainer.py            # Trains LightGBM with Optuna hyperparameters & dynamic thresholds
│   │
│   └── pipeline/                       
│       ├── train_pipeline.py           # Orchestrates the full training lifecycle
│       └── predict_pipeline.py         # Transforms incoming API payloads using the feature store
│
├── app.py                              # FastAPI deployment script and Uvicorn server configuration
├── Dockerfile                          # Containerization instructions for cloud deployment
└── requirements.txt                    # Python dependencies
