"""
Instamart Capstone — Part B: Customer Segmentation and Lapse Prediction

Run from the project root:
    python src/part_b/part_b.py

Uses only the cleaned Part A CSV files and allowed libraries.
The lapse model uses orders strictly before 2025-09-01 as features.
The target is whether a customer has no completed order on/after 2025-09-01.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
CUTOFF = pd.Timestamp("2025-09-01")


def load_inputs():
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    products = pd.read_csv(CLEAN / "products.csv")
    required = {"customer_id", "order_id", "order_ts", "status", "order_amount",
                "quantity", "product_id", "discount_pct"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders.csv is missing required columns: {sorted(missing)}")
    return customers, orders, products


def completed_orders(orders):
    return orders.loc[
        orders["status"].astype(str).str.strip().str.lower().eq("completed")
    ].copy()


def make_rfm(customers, orders):
    """Whole-observation-window RFM for descriptive segmentation only."""
    done = completed_orders(orders)
    reference_date = done["order_ts"].max().normalize() + pd.Timedelta(days=1)
    rfm = done.groupby("customer_id").agg(
        recency_days=("order_ts", lambda s: int((reference_date - s.max().normalize()).days)),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
    ).reset_index()
    rfm = customers[["customer_id"]].merge(rfm, on="customer_id", how="left")
    rfm[["recency_days", "frequency", "monetary", "average_order_value"]] = (
        rfm[["recency_days", "frequency", "monetary", "average_order_value"]].fillna(0)
    )
    return rfm, reference_date


def segment_customers(customers, rfm):
    # Log transforms reduce the influence of highly skewed spending/frequency.
    features = rfm[["recency_days", "frequency", "monetary"]].copy()
    features["recency_days"] = np.log1p(features["recency_days"].clip(lower=0))
    features["frequency"] = np.log1p(features["frequency"].clip(lower=0))
    features["monetary"] = np.log1p(features["monetary"].clip(lower=0))
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_clusters = min(4, max(2, len(rfm)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X)
    result = rfm.copy()
    result["segment"] = labels
    result["pca_1"] = coordinates[:, 0]
    result["pca_2"] = coordinates[:, 1]

    # Human-readable cluster descriptions from the cluster-level RFM profile.
    profile = result.groupby("segment")[["recency_days", "frequency", "monetary"]].mean()
    overall = result[["recency_days", "frequency", "monetary"]].mean()
    descriptions = {}
    for cluster, row in profile.iterrows():
        if row["frequency"] >= overall["frequency"] and row["monetary"] >= overall["monetary"]:
            name = "High-value active"
        elif row["recency_days"] > overall["recency_days"] and row["frequency"] < overall["frequency"]:
            name = "At-risk / low-engagement"
        elif row["frequency"] >= overall["frequency"]:
            name = "Frequent customers"
        else:
            name = "Developing customers"
        descriptions[int(cluster)] = name
    result["segment_name"] = result["segment"].map(descriptions)
    return result, scaler, kmeans, pca, descriptions


def make_cohorts(customers, orders):
    """Signup-month cohorts with completed-order customer activity by order month."""
    c = customers[["customer_id", "signup_date"]].copy()
    c["cohort_month"] = c["signup_date"].dt.to_period("M").astype(str)
    done = completed_orders(orders)
    done["order_month"] = done["order_ts"].dt.to_period("M").astype(str)
    joined = done.merge(c[["customer_id", "cohort_month"]], on="customer_id", how="inner")
    cohort = (joined.groupby(["cohort_month", "order_month"])["customer_id"]
              .nunique().rename("active_customers").reset_index())
    cohort["cohort_index"] = (
        pd.PeriodIndex(cohort["order_month"], freq="M").astype(int)
        - pd.PeriodIndex(cohort["cohort_month"], freq="M").astype(int) + 1
    )
    cohort.to_csv(OUTPUTS / "cohort_analysis.csv", index=False)
    return cohort


def build_lapse_features(customers, orders, products):
    """Build pre-cutoff features and a post-cutoff inactivity target."""
    done = completed_orders(orders)
    before = done.loc[done["order_ts"] < CUTOFF].copy()
    after = done.loc[done["order_ts"] >= CUTOFF].copy()

    # Only pre-cutoff transactions contribute to model features.
    if before.empty:
        raise ValueError("No completed orders exist before 2025-09-01; cannot build lapse features.")

    features = before.groupby("customer_id").agg(
        feature_frequency=("order_id", "nunique"),
        feature_monetary=("order_amount", "sum"),
        feature_avg_order_value=("order_amount", "mean"),
        feature_avg_discount=("discount_pct", "mean"),
        feature_total_quantity=("quantity", "sum"),
        feature_unique_products=("product_id", "nunique"),
        feature_last_order=("order_ts", "max"),
    ).reset_index()
    features["feature_recency_days"] = (
        CUTOFF - features["feature_last_order"].dt.normalize()
    ).dt.days
    features = features.drop(columns=["feature_last_order"])

    # Add pre-cutoff category variety using the product catalogue.
    if "category" in products.columns:
        product_category = products[["product_id", "category"]].drop_duplicates("product_id")
        category_counts = (before.merge(product_category, on="product_id", how="left")
                           .groupby("customer_id")["category"].nunique()
                           .rename("feature_category_diversity").reset_index())
        features = features.merge(category_counts, on="customer_id", how="left")
    else:
        features["feature_category_diversity"] = 0

    # Customers with no pre-cutoff completed order receive zero-valued history features.
    all_customers = customers[["customer_id", "city", "gender", "age", "signup_date"]].copy()
    data = all_customers.merge(features, on="customer_id", how="left")
    numeric_history = [c for c in data.columns if c.startswith("feature_")]
    data[numeric_history] = data[numeric_history].fillna(0)
    data["tenure_days_at_cutoff"] = (CUTOFF - data["signup_date"]).dt.days.clip(lower=0)
    data = data.drop(columns=["signup_date"])

    post_counts = after.groupby("customer_id")["order_id"].nunique()
    data["post_cutoff_orders"] = data["customer_id"].map(post_counts).fillna(0).astype(int)
    data["lapsed"] = (data["post_cutoff_orders"] == 0).astype(int)
    return data


def train_lapse_model(data):
    # Exclude IDs, post-cutoff outcome/counts, and any whole-window segmentation data.
    target = data["lapsed"].astype(int)
    excluded = {"customer_id", "post_cutoff_orders", "lapsed"}
    X = data[[c for c in data.columns if c not in excluded]].copy()

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

    counts = target.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError(
            "The lapse target does not have at least 5 examples in each class. "
            f"Target counts: {counts.to_dict()}. Check the dataset date range and target definition."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.25, random_state=42, stratify=target
    )
    cv_splits = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [None, 8, 16],
             "model__min_samples_leaf": [2, 5]},
        ),
    }
    results = {}
    best_name, best_search, best_f1 = None, None, -1
    for name, (estimator, grid) in models.items():
        pipe = Pipeline([("preprocess", preprocessing), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        score = f1_score(y_test, prediction, zero_division=0)
        results[name] = {
            "best_cv_f1": float(search.best_score_),
            "test_f1": float(score),
            "best_params": search.best_params_,
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
            "classification_report": classification_report(
                y_test, prediction, output_dict=True, zero_division=0
            ),
        }
        print(f"\n{name}: best CV F1={search.best_score_:.3f}; test F1={score:.3f}")
        print(classification_report(y_test, prediction, zero_division=0))
        if score > best_f1:
            best_name, best_search, best_f1 = name, search, score

    model_path = OUTPUTS / "lapse_model.joblib"
    joblib.dump({
        "model": best_search.best_estimator_,
        "model_name": best_name,
        "feature_columns": X.columns.tolist(),
        "cutoff": str(CUTOFF.date()),
        "target_definition": "1 = no completed order on/after 2025-09-01",
        "test_f1": float(best_f1),
        "all_model_results": results,
    }, model_path)

    # Score every customer using the selected pre-cutoff model.
    probabilities = best_search.best_estimator_.predict_proba(X)[:, 1]
    scored = data[["customer_id"]].copy()
    scored["predicted_lapse_probability"] = probabilities
    scored["predicted_lapse"] = (probabilities >= 0.5).astype(int)
    scored["actual_lapsed_for_evaluation"] = target.values
    return scored, results, best_name, best_f1


def main():
    customers, orders, products = load_inputs()
    rfm, reference_date = make_rfm(customers, orders)
    segments, segment_scaler, kmeans, pca, descriptions = segment_customers(customers, rfm)
    segments.to_csv(OUTPUTS / "customer_segments.csv", index=False)
    make_cohorts(customers, orders)

    model_data = build_lapse_features(customers, orders, products)
    scored, model_results, best_name, best_f1 = train_lapse_model(model_data)

    # The premium flag is descriptive/post-hoc and uses whole-window monetary RFM.
    # It is not a model feature and must not be interpreted as leakage-free evaluation.
    premium_cutoff = segments["monetary"].quantile(0.75)
    premium = segments[["customer_id", "monetary"]].copy()
    premium["premium_customer"] = premium["monetary"] >= premium_cutoff
    retention = scored.merge(premium[["customer_id", "premium_customer", "monetary"]],
                             on="customer_id", how="left")
    retention = retention.sort_values(
        ["predicted_lapse_probability", "premium_customer", "monetary"],
        ascending=[False, False, False]
    )
    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)

    pd.DataFrame([
        {"model": name, "best_cv_f1": result["best_cv_f1"],
         "test_f1": result["test_f1"], "best_params": str(result["best_params"])}
        for name, result in model_results.items()
    ]).to_csv(OUTPUTS / "lapse_model_comparison.csv", index=False)

    print("\nPART B COMPLETE")
    print(f"RFM reference date: {reference_date.date()}")
    print(f"Segments saved: {OUTPUTS / 'customer_segments.csv'}")
    print(f"Cohort analysis saved: {OUTPUTS / 'cohort_analysis.csv'}")
    print(f"Best lapse model: {best_name}; test F1={best_f1:.3f}")
    print(f"Model saved: {OUTPUTS / 'lapse_model.joblib'}")
    print(f"Retention list saved: {OUTPUTS / 'retention_list.csv'}")
    print("\nNote: segmentation uses the full observation window for descriptive analysis.")
    print("The lapse model uses only pre-2025-09-01 features; premium status is post-hoc.")


if __name__ == "__main__":
    main()
