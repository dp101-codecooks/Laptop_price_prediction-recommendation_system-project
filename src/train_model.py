# src/train_model.py
import os
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/laptops.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "laptop_price_model.pkl"

def ensure_paths():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place your CSV as data/laptops.csv and re-run."
        )
    df = pd.read_csv(DATA_PATH)
    # Normalize column names (lowercase, no spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# --- Helpers to parse messy specs -----------------
def parse_ram(val):
    # e.g. "8GB" -> 8
    if pd.isna(val): return np.nan
    s = str(val).upper().replace(" ", "")
    m = re.search(r"(\d+)\s*GB", s)
    return float(m.group(1)) if m else pd.to_numeric(val, errors="coerce")

def parse_weight(val):
    # e.g. "1.37kg" -> 1.37
    if pd.isna(val): return np.nan
    s = str(val).lower().replace(" ", "")
    s = s.replace("kgs", "kg")
    if s.endswith("kg"):
        s = s[:-2]
    return pd.to_numeric(s, errors="coerce")

def to_gb(value):
    # "1TB" -> 1024, "512GB" -> 512
    s = str(value).upper().replace(" ", "")
    if "TB" in s:
        m = re.search(r"(\d+\.?\d*)TB", s)
        return float(m.group(1)) * 1024 if m else np.nan
    m = re.search(r"(\d+\.?\d*)GB", s)
    return float(m.group(1)) if m else pd.to_numeric(value, errors="coerce")

def parse_memory(val):
    """
    Convert 'Memory' like '512GB SSD + 1TB HDD' -> total GB (e.g., 1536)
    If dataset already has numeric storage columns, we’ll handle that below.
    """
    if pd.isna(val): return np.nan
    parts = re.split(r"\+|,", str(val))
    total = 0.0
    for p in parts:
        gb = to_gb(p)
        if gb and not np.isnan(gb):
            total += gb
    return total if total > 0 else pd.to_numeric(val, errors="coerce")

def extract_cpu_brand(cpu_str):
    if pd.isna(cpu_str): return "Other"
    s = str(cpu_str).upper()
    if "INTEL" in s: return "Intel"
    if "RYZEN" in s or "AMD" in s: return "AMD"
    if "APPLE" in s or "M1" in s or "M2" in s or "M3" in s: return "Apple"
    return "Other"

def extract_gpu_brand(gpu_str):
    if pd.isna(gpu_str): return "Other"
    s = str(gpu_str).upper()
    if "NVIDIA" in s or "GEFORCE" in s or "RTX" in s or "GTX" in s: return "NVIDIA"
    if "AMD" in s or "RADEON" in s: return "AMD"
    if "INTEL" in s: return "Intel"
    return "Other"

def pick_target_column(df):
    # Try common target names
    for name in ["price", "price_rs", "price_inr", "price_euros", "price_usd"]:
        if name in df.columns:
            return name
    raise KeyError(
        "Price/target column not found. Rename your target to one of: "
        "'price', 'price_rs', 'price_inr', 'price_euros', 'price_usd'."
    )

def engineer_features(df):
    df = df.copy()

    # Attempt to locate common columns; fallbacks allowed
    # Company / Brand
    if "company" in df.columns:
        df["brand"] = df["company"]
    elif "brand" in df.columns:
        pass
    else:
        df["brand"] = "Unknown"

    # CPU / GPU
    if "cpu" not in df.columns:
        df["cpu"] = df.get("processor", df.get("processor_brand", "Unknown"))
    if "gpu" not in df.columns:
        df["gpu"] = df.get("graphics_card", df.get("graphics", "Unknown"))

    # RAM
    if "ram" in df.columns:
        df["ram_gb"] = df["ram"].apply(parse_ram)
    else:
        df["ram_gb"] = pd.to_numeric(df.get("ram_gb", np.nan), errors="coerce")

    # Weight
    if "weight" in df.columns:
        df["weight_kg"] = df["weight"].apply(parse_weight)
    else:
        df["weight_kg"] = pd.to_numeric(df.get("weight_kg", np.nan), errors="coerce")

    # Inches / Screen size
    if "inches" in df.columns:
        df["inches"] = pd.to_numeric(df["inches"], errors="coerce")
    else:
        df["inches"] = pd.to_numeric(df.get("screen_size", np.nan), errors="coerce")

    # ScreenResolution (keep as categorical text)
    if "screenresolution" in df.columns:
        df["screenresolution"] = df["screenresolution"].astype(str)
    elif "screen_resolution" in df.columns:
        df["screenresolution"] = df["screen_resolution"].astype(str)
    else:
        df["screenresolution"] = df.get("resolution", "Unknown").astype(str)

    # OpSys / OS
    if "opsys" in df.columns:
        df["opsys"] = df["opsys"].astype(str)
    else:
        df["opsys"] = df.get("os", "Unknown").astype(str)

    # Memory (total GB)
    if "memory" in df.columns:
        df["total_storage_gb"] = df["memory"].apply(parse_memory)
    else:
        # try storage columns
        storage_candidates = [c for c in df.columns if "storage" in c or "ssd" in c or "hdd" in c]
        if storage_candidates:
            total = np.zeros(len(df), dtype=float)
            for c in storage_candidates:
                total += df[c].apply(to_gb).fillna(0).values
            df["total_storage_gb"] = total
        else:
            df["total_storage_gb"] = np.nan

    # CPU/GPU brand
    df["cpu_brand"] = df["cpu"].apply(extract_cpu_brand)
    df["gpu_brand"] = df["gpu"].apply(extract_gpu_brand)

    # Minimal, robust feature set
    feature_cols = [
        "brand", "cpu_brand", "gpu_brand", "opsys", "screenresolution",
        "inches", "ram_gb", "total_storage_gb", "weight_kg"
    ]
    # Ensure all exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[feature_cols], feature_cols

def build_pipeline(numeric_features, categorical_features):
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    return pipe

def main():
    ensure_paths()
    df = load_dataset()

    target_col = pick_target_column(df)
    Xdf, feature_cols = engineer_features(df)

    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = ~y.isna()
    Xdf = Xdf.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    numeric_features = ["inches", "ram_gb", "total_storage_gb", "weight_kg"]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        Xdf, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"R2  : {r2:.4f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model pipeline → {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
