"""
Phase 1 – Part 1
Generate 10,000 synthetic patient records and load them into SQLite.
No real patient data used — purely educational.
"""

import sqlite3
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
DB_PATH  = os.path.join(os.path.dirname(__file__), "../database/healthcare.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "patients.csv")
N        = 10_000

# ── Lookup tables ─────────────────────────────────────────────────────────────
FIRST_NAMES  = ["Alice","Bob","Carol","David","Emma","Frank","Grace","Henry",
                 "Iris","James","Karen","Leo","Maria","Nathan","Olivia","Paul",
                 "Quinn","Rachel","Sam","Tina","Uma","Victor","Wendy","Xander",
                 "Yara","Zoe","Arjun","Priya","Raj","Sneha","Mihir","Ananya"]
LAST_NAMES   = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
                 "Davis","Wilson","Moore","Taylor","Anderson","Thomas","Jackson",
                 "White","Harris","Martin","Thompson","Lee","Patel","Shah","Khan"]
GENDERS      = ["Male","Female","Other"]
BLOOD_TYPES  = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
CONDITIONS   = ["Diabetes","Hypertension","Asthma","Arthritis","Cancer",
                "Heart Disease","Obesity","Pneumonia","Anxiety","Depression",
                "COVID-19","Migraine","Kidney Disease","Anemia","Thyroid Disorder"]
MEDICATIONS  = ["Metformin","Lisinopril","Albuterol","Ibuprofen","Paracetamol",
                "Atorvastatin","Omeprazole","Aspirin","Amlodipine","Levothyroxine",
                "Insulin","Prednisone","Amoxicillin","Cetirizine","Sertraline"]
TEST_RESULTS = ["Normal","Abnormal","Inconclusive"]
HOSPITALS    = ["City General Hospital","St. Mary Medical Center",
                "Sunrise Health Institute","Valley Care Hospital",
                "Metro Emergency Clinic","Hillside Community Hospital"]
DOCTORS      = ["Dr. Patel","Dr. Smith","Dr. Johnson","Dr. Lee","Dr. Garcia",
                "Dr. Martinez","Dr. Wilson","Dr. Anderson","Dr. Thompson","Dr. Brown"]
ADMISSION_TYPES   = ["Emergency","Elective","Urgent"]
INSURANCE_PROVIDERS = ["BlueCross","Aetna","UnitedHealth","Cigna","Humana","Medicare","Medicaid"]

def rand_date(start="2022-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    return s + timedelta(days=random.randint(0, (e - s).days))

def generate_patients(n=N):
    records = []
    for i in range(1, n + 1):
        admission = rand_date()
        los = random.randint(1, 30)          # length of stay in days
        discharge = admission + timedelta(days=los)
        billing = round(random.uniform(500, 50000), 2)
        records.append({
            "patient_id"        : i,
            "name"              : f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "age"               : random.randint(1, 95),
            "gender"            : random.choice(GENDERS),
            "blood_type"        : random.choice(BLOOD_TYPES),
            "medical_condition" : random.choice(CONDITIONS),
            "medication"        : random.choice(MEDICATIONS),
            "test_results"      : random.choice(TEST_RESULTS),
            "hospital"          : random.choice(HOSPITALS),
            "doctor"            : random.choice(DOCTORS),
            "room_number"       : random.randint(100, 999),
            "admission_type"    : random.choice(ADMISSION_TYPES),
            "admission_date"    : admission.strftime("%Y-%m-%d"),
            "discharge_date"    : discharge.strftime("%Y-%m-%d"),
            "insurance_provider": random.choice(INSURANCE_PROVIDERS),
            "billing_amount"    : billing,
        })
    return pd.DataFrame(records)

# ── Build SQLite database ─────────────────────────────────────────────────────
def build_database(df: pd.DataFrame, db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── patients (core demographics) ─────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   INTEGER PRIMARY KEY,
            name         TEXT NOT NULL,
            age          INTEGER,
            gender       TEXT,
            blood_type   TEXT
        )
    """)

    # ── clinical_data ─────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clinical_data (
            record_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id        INTEGER REFERENCES patients(patient_id),
            medical_condition TEXT,
            medication        TEXT,
            test_results      TEXT
        )
    """)

    # ── admissions ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admissions (
            admission_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER REFERENCES patients(patient_id),
            hospital        TEXT,
            doctor          TEXT,
            room_number     INTEGER,
            admission_type  TEXT,
            admission_date  TEXT,
            discharge_date  TEXT
        )
    """)

    # ── billing ───────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS billing (
            billing_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id         INTEGER REFERENCES patients(patient_id),
            insurance_provider TEXT,
            billing_amount     REAL
        )
    """)

    # ── indexes ───────────────────────────────────────────────────────────────
    cur.execute("CREATE INDEX IF NOT EXISTS idx_patient_condition ON clinical_data(medical_condition)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_admission_date    ON admissions(admission_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_test_results      ON clinical_data(test_results)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hospital          ON admissions(hospital)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_insurance         ON billing(insurance_provider)")

    conn.commit()

    # ── Insert data ───────────────────────────────────────────────────────────
    df[["patient_id","name","age","gender","blood_type"]]\
        .to_sql("patients", conn, if_exists="replace", index=False)

    df[["patient_id","medical_condition","medication","test_results"]]\
        .to_sql("clinical_data", conn, if_exists="replace", index=False)

    df[["patient_id","hospital","doctor","room_number",
        "admission_type","admission_date","discharge_date"]]\
        .to_sql("admissions", conn, if_exists="replace", index=False)

    df[["patient_id","insurance_provider","billing_amount"]]\
        .to_sql("billing", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print(f"  ✓ Database created at {db_path}")
    print(f"  ✓ Tables: patients, clinical_data, admissions, billing")

# ── Validation ────────────────────────────────────────────────────────────────
def validate_database(db_path: str):
    conn = sqlite3.connect(db_path)
    print("\n  ── Validation Report ─────────────────────────")
    for table in ["patients","clinical_data","admissions","billing"]:
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn)["n"][0]
        nulls = pd.read_sql(f"SELECT * FROM {table} LIMIT 1000", conn).isnull().sum().sum()
        print(f"    {table:15s}: {count:6,} rows | nulls in sample: {nulls}")
    conn.close()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[Phase 1 – Part 1] Generating synthetic healthcare dataset...")
    df = generate_patients()
    df.to_csv(CSV_PATH, index=False)
    print(f"  ✓ CSV saved  → {CSV_PATH}")
    build_database(df, DB_PATH)
    validate_database(DB_PATH)
    print("\n  Done.\n")