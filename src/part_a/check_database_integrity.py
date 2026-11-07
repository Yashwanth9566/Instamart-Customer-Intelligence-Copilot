import sqlite3
from pathlib import Path

DB = Path("data/instamart.db")

conn = sqlite3.connect(DB)
cur = conn.cursor()

checks = {
    "Orders without customers": """
        SELECT COUNT(*)
        FROM orders o
        LEFT JOIN customers c
        ON o.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """,
    "Orders without products": """
        SELECT COUNT(*)
        FROM orders o
        LEFT JOIN products p
        ON o.product_id = p.product_id
        WHERE p.product_id IS NULL
    """
}

print("DATABASE INTEGRITY CHECK")

for label, query in checks.items():
    result = cur.execute(query).fetchone()[0]
    print(f"{label}: {result}")

print("\nForeign key check:")
print(cur.execute("PRAGMA foreign_key_check").fetchall())

conn.close()