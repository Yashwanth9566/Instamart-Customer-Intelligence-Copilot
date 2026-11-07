"""
Instamart Capstone — Part A: Data Foundation
Run from the project root:
    python src/part_a/part_a.py

This script audits the raw exports, cleans customers/products/orders,
writes clean CSVs, creates the SQLite database, and validates key checkpoints.
"""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"
OUTPUTS = ROOT / "outputs"
CLEAN.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

def audit_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = df.isna().sum()
    report = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(missing[c]) for c in df.columns],
        "missing_percent": [round(float(missing[c]) / len(df) * 100, 2) for c in df.columns],
    })
    print(f"\n{name}: shape={df.shape}")
    print(report.to_string(index=False))
    return report

def main():
    # Read raw data. Keep the original files unchanged.
    customers_raw = pd.read_csv(RAW / "customers.csv", dtype=str, keep_default_na=False)
    products_raw = pd.read_csv(RAW / "products.csv")
    orders_raw = pd.read_csv(RAW / "orders.csv", dtype=str, keep_default_na=False)

    audit_table(customers_raw.replace("", pd.NA), "RAW CUSTOMERS")
    audit_table(products_raw, "RAW PRODUCTS")
    audit_table(orders_raw.replace("", pd.NA), "RAW ORDERS")

    # Detailed raw audit figures required by the brief.
    amount_numeric = pd.to_numeric(orders_raw["order_amount"].replace("", pd.NA), errors="coerce")
    age_numeric = pd.to_numeric(customers_raw["age"].replace("", pd.NA), errors="coerce")
    audit = {
        "raw_order_rows": len(orders_raw),
        "exact_duplicate_order_rows": int(orders_raw.duplicated().sum()),
        "pending_rows": int(orders_raw["status"].str.strip().str.lower().eq("pending").sum()),
        "distinct_order_ids": int(orders_raw["order_id"].nunique()),
        "non_numeric_order_amounts": int(amount_numeric.isna().sum()),
        "negative_delivery_minutes": int((pd.to_numeric(orders_raw["delivery_minutes"].replace("", pd.NA), errors="coerce") < 0).sum()),
        "blank_delivery_minutes": int(orders_raw["delivery_minutes"].str.strip().eq("").sum()),
        "blank_customer_ages": int(customers_raw["age"].str.strip().eq("").sum()),
        "ages_outside_10_100": int(((age_numeric < 10) | (age_numeric > 100)).sum()),
        "distinct_raw_city_spellings": int(customers_raw["city"].str.strip().nunique()),
        "distinct_nonblank_raw_gender_spellings": int(customers_raw.loc[customers_raw["gender"].str.strip().ne(""), "gender"].str.strip().nunique()),
        "blank_genders": int(customers_raw["gender"].str.strip().eq("").sum()),
    }
    pd.DataFrame([audit]).to_csv(OUTPUTS / "part_a_raw_audit.csv", index=False)
    print("\nRAW AUDIT FIGURES")
    for key, value in audit.items():
        print(f"{key}: {value}")

    # Clean customers.
    customers = customers_raw.copy()
    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    customers["gender"] = customers["gender"].astype(str).str.strip().str.lower()
    customers["gender"] = customers["gender"].replace({
        "m": "male", "male": "male",
        "f": "female", "female": "female",
        "": "unknown",
    })
    customers.loc[~customers["gender"].isin(["male", "female"]), "gender"] = "unknown"
    customers["age"] = pd.to_numeric(customers["age"].replace("", pd.NA), errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA
    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )
    # Defensive fallback in case a city has no valid ages.
    customers["age"] = customers["age"].fillna(customers["age"].median()).round().astype(int)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")

    # Clean products: preserve product attributes, standardise text fields and numeric price.
    products = products_raw.copy()
    for col in ["product_id", "product_name", "category"]:
        products[col] = products[col].astype(str).str.strip()
    products["category"] = products["category"].str.title()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")
    if products["price_inr"].isna().any() or (products["price_inr"] <= 0).any():
        raise ValueError("Products contain missing, non-numeric, or non-positive prices.")

    # Clean orders in the exact order specified by the assignment.
    orders = orders_raw.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders = orders.drop_duplicates()
    orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates(
        subset=["order_id"], keep="last"
    ).copy()
    for col in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
        orders[col] = pd.to_numeric(orders[col].replace("", pd.NA), errors="coerce")
    orders = orders.merge(products[["product_id", "category", "price_inr"]],
                          on="product_id", how="left", validate="many_to_one")
    orders["order_amount"] = orders["order_amount"].fillna(
        (orders["price_inr"] * orders["quantity"] *
         (1 - orders["discount_pct"] / 100)).round(2)
    )
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA
    completed = orders["status"].str.strip().str.lower().eq("completed")
    category_medians = orders.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    # Map the completed-order category median back to each completed row.
    medians_by_category = orders.loc[completed].groupby("category")["delivery_minutes"].median()
    missing_completed = completed & orders["delivery_minutes"].isna()
    orders.loc[missing_completed, "delivery_minutes"] = orders.loc[missing_completed, "category"].map(medians_by_category)
    # Cancelled orders intentionally retain missing delivery times.
    orders = orders.drop(columns=["category", "price_inr"])
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save clean CSVs.
    customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)

    # SQLite tables.
    with sqlite3.connect(DB) as conn:
        customers_db = customers.copy()
        customers_db["signup_date"] = customers_db["signup_date"].dt.strftime("%Y-%m-%d")
        customers_db.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)

    # Reference validations. Fail loudly rather than silently accepting wrong data.
    completed_orders = orders[orders["status"].str.lower().eq("completed")]
    cancelled_orders = orders[orders["status"].str.lower().eq("cancelled")]
    checks = [
        len(customers) == 1200,
        customers["city"].nunique() == 6,
        customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
        customers["age"].isna().sum() == 0,
        customers["age"].between(18, 60).all(),
        len(orders) == 12309,
        orders["order_id"].nunique() == 12309,
        int(orders["status"].str.lower().eq("completed").sum()) == 11110,
        int(orders["status"].str.lower().eq("cancelled").sum()) == 1199,
        not orders["status"].str.lower().eq("pending").any(),
        orders["order_amount"].isna().sum() == 0,
        completed_orders["delivery_minutes"].isna().sum() == 0,
        cancelled_orders["delivery_minutes"].isna().sum() == 1199,
        abs(completed_orders["order_amount"].sum() - 2998668.80) < 0.01,
    ]
    if not all(checks):
        print("\nValidation checks:", [bool(x) for x in checks])
        raise AssertionError("One or more Part A reference checkpoints failed. See audit and cleaning logic.")
    print("\nALL CHECKPOINTS PASSED")
    print(f"Clean files saved to: {CLEAN}")
    print(f"SQLite database saved to: {DB}")

if __name__ == "__main__":
    main()
