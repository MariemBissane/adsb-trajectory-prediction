"""
pipeline.py — Pipeline automatique ML + DL — Prédiction de Trajectoires ADS-B
==============================================================================
Aligné sur le notebook 03_ml_models_multi_horizons :
  - Multi-horizons : 2, 5, 10 minutes
  - Régression par deltas (delta_lat, delta_lon, delta_altitude)
  - Features enrichies (vx_ms, vy_ms, heading_sin/cos, distance_step_m, …)
  - Baseline naïve + Decision Tree + Random Forest + Gradient Boosting
  - Optimisation Optuna (Random Forest)
  - SVM RBF avec PCA (classification d'intention)
  - LabelEncoder pour les classes
  - evaluate_regression_model avec p95 et p99
  - Résidus + comparaison multi-horizons
  - DL inchangé (GRU / LSTM / TCN)

Usage :
    python pipeline.py --data_dir data/raw --mode both
    python pipeline.py --data_dir data/raw --mode ml
    python pipeline.py --data_dir data/raw --mode dl --horizon 5
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse, time, warnings, pickle, gc, sys, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

HORIZONS_MINUTES  = [2, 5, 10]   # tous les horizons traités
RESAMPLING_SECONDS = 10


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/raw")
    p.add_argument("--horizon",    type=int, default=None,
                   help="Horizon unique (2/5/10 min). Si absent, tous les horizons sont traités.")
    p.add_argument("--mode",       default="both", choices=["ml", "dl", "both"])
    p.add_argument("--output_dir", default="results/pipeline")
    p.add_argument("--n_traj",     type=int, default=2000,
                   help="Nombre max de trajectoires à charger pour le ML.")
    p.add_argument("--optuna_trials", type=int, default=10,
                   help="Nombre d'essais Optuna pour le tuning RF.")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
USEFUL_COLUMNS = [
    "time", "icao24", "lat", "lon",
    "velocity", "heading", "vertrate",
    "baroaltitude", "geoaltitude"
]

def load_raw_data(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(
        list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz"))
    )
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV dans {data_dir}")
    log(f"  {len(csv_files)} fichier(s) détecté(s)")

    frames = []
    for f in csv_files:
        cols_dispo = pd.read_csv(f, nrows=1).columns.tolist()
        cols = [c for c in USEFUL_COLUMNS if c in cols_dispo]
        df = pd.read_csv(f, usecols=cols)
        df["source_file"] = f.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — NETTOYAGE
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    for col in ["time","lat","lon","velocity","heading","vertrate","baroaltitude","geoaltitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["icao24"] = df["icao24"].astype("string").str.strip().str.lower()
    df = df.dropna(subset=["time", "icao24", "lat", "lon"])
    df = df[df["icao24"] != ""]
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.dropna(subset=["velocity", "heading", "vertrate"])

    if "geoaltitude" in df.columns and "baroaltitude" in df.columns:
        df["altitude"] = df["geoaltitude"].fillna(df["baroaltitude"])
        df = df.drop(columns=["geoaltitude", "baroaltitude"])
    elif "baroaltitude" in df.columns:
        df = df.rename(columns={"baroaltitude": "altitude"})
    elif "geoaltitude" in df.columns:
        df = df.rename(columns={"geoaltitude": "altitude"})

    df = df.dropna(subset=["altitude"])
    df = df[
        df["lat"].between(-90, 90) &
        df["lon"].between(-180, 180) &
        (df["velocity"] >= 0) &
        df["heading"].between(0, 360) &
        df["altitude"].between(-500, 20000)
    ]
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["icao24", "datetime"], keep="first")
    df = df.sort_values(["icao24", "datetime"]).reset_index(drop=True)

    log(f"  Nettoyage : {n0:,} -> {len(df):,} lignes ({n0-len(df):,} supprimées)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — RESAMPLING + FEATURES (aligné notebook 03)
# ─────────────────────────────────────────────────────────────────────────────
def add_features_one_trajectory(traj: pd.DataFrame, rolling_window=5) -> pd.DataFrame:
    """
    Feature engineering aligné sur les colonnes utilisées dans le notebook 03 :
    - cinématique : ground_speed_ms, vertical_rate_fpm, acceleration, turn_rate
    - direction   : vx_ms, vy_ms, vz_ms, heading_sin, heading_cos
    - géométrique : distance_step_m, distance_traveled_m, trajectory_angle,
                    distance_destination_m, elapsed_time_sec
    - rolling     : roll_mean/std sur velocity, altitude, vertrate, acceleration,
                    turn_rate, distance_step_m
    - phase de vol estimée
    """
    traj = traj.copy().reset_index(drop=True)
    traj["dt_sec"] = traj["time"].diff().fillna(1).replace(0, 1)

    # ── Cinématique ──────────────────────────────────────────────────────────
    traj["ground_speed_ms"] = traj["velocity"]                     # velocity en m/s
    traj["vertical_rate_fpm"] = traj["vertrate"] * 196.85          # m/s → ft/min
    traj["acceleration"] = traj["velocity"].diff().fillna(0) / traj["dt_sec"]

    heading_diff = traj["heading"].diff().fillna(0)
    heading_diff = ((heading_diff + 180) % 360) - 180
    traj["turn_rate"] = heading_diff / traj["dt_sec"]

    # ── Direction ────────────────────────────────────────────────────────────
    heading_rad = np.radians(traj["heading"])
    traj["heading_sin"] = np.sin(heading_rad)
    traj["heading_cos"] = np.cos(heading_rad)

    traj["vx_ms"] = traj["velocity"] * np.sin(heading_rad)
    traj["vy_ms"] = traj["velocity"] * np.cos(heading_rad)
    traj["vz_ms"] = traj["vertrate"]

    # ── Géométrique ──────────────────────────────────────────────────────────
    lat_rad  = np.radians(traj["lat"])
    lon_rad  = np.radians(traj["lon"])
    dlat     = lat_rad.diff().fillna(0)
    dlon     = lon_rad.diff().fillna(0)
    R        = 6371000.0
    a        = np.sin(dlat/2)**2 + np.cos(lat_rad.shift(1).fillna(lat_rad)) * np.cos(lat_rad) * np.sin(dlon/2)**2
    traj["distance_step_m"]     = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    traj["distance_traveled_m"] = traj["distance_step_m"].cumsum()
    traj["elapsed_time_sec"]    = (traj["time"] - traj["time"].iloc[0]).clip(lower=0)

    # Angle de trajectoire
    traj["trajectory_angle"] = np.degrees(
        np.arctan2(
            traj["lat"].diff().fillna(0),
            traj["lon"].diff().fillna(0)
        )
    )

    # Distance à la destination (dernier point)
    dest_lat = traj["lat"].iloc[-1]
    dest_lon = traj["lon"].iloc[-1]
    a_dest = (
        np.sin(np.radians((traj["lat"] - dest_lat) / 2))**2
        + np.cos(np.radians(traj["lat"])) * np.cos(np.radians(dest_lat))
        * np.sin(np.radians((traj["lon"] - dest_lon) / 2))**2
    )
    traj["distance_destination_m"] = 2 * R * np.arcsin(np.sqrt(np.clip(a_dest, 0, 1)))

    # ── Rolling features ─────────────────────────────────────────────────────
    rolling_targets = {
        "velocity":       "roll_{stat}_velocity",
        "altitude":       "roll_{stat}_altitude",
        "vertrate":       "roll_{stat}_vertrate",
        "acceleration":   "roll_{stat}_acceleration",
        "turn_rate":      "roll_{stat}_turn_rate",
        "distance_step_m":"roll_{stat}_distance_step_m",
    }
    for col, tmpl in rolling_targets.items():
        r = traj[col].rolling(window=rolling_window, min_periods=1)
        traj[tmpl.format(stat="mean")] = r.mean()
        traj[tmpl.format(stat="std")]  = r.std().fillna(0)

    # ── Phase de vol ─────────────────────────────────────────────────────────
    conditions = [
        traj["vertrate"] > 2,
        traj["vertrate"] < -2,
        traj["turn_rate"].abs() > 2,
    ]
    traj["flight_phase"] = np.select(conditions,
                                     ["climbing", "descending", "turning"],
                                     default="cruise")
    phase_map = {"climbing": 0, "descending": 1, "turning": 2, "cruise": 3}
    traj["flight_phase_enc"] = traj["flight_phase"].map(phase_map)

    return traj


def resample_and_features(df: pd.DataFrame, freq_sec: int = RESAMPLING_SECONDS) -> pd.DataFrame:
    results = []
    for icao, group in df.groupby("icao24"):
        group = group.sort_values("datetime").reset_index(drop=True)
        group["time_rounded"] = (group["time"] // freq_sec) * freq_sec
        group = group.drop_duplicates(subset=["time_rounded"]).copy()
        group["time"] = group["time_rounded"]
        if len(group) < 10:
            continue
        feat = add_features_one_trajectory(group)
        feat["trajectory_id"] = icao
        results.append(feat)

    if not results:
        raise ValueError("Aucune trajectoire valide après resampling.")

    out = pd.concat(results, ignore_index=True)
    log(f"  Features : {out['trajectory_id'].nunique()} trajectoires, {len(out):,} points")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — TARGETS MULTI-HORIZONS (notebook 03)
# ─────────────────────────────────────────────────────────────────────────────
def create_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les targets pour les 3 horizons (2, 5, 10 min) dans le même DataFrame.
    Colonnes produites : lat_future_Xmin, lon_future_Xmin,
                         altitude_future_Xmin, intent_future_Xmin
    """
    df = df.sort_values(["trajectory_id", "time"]).reset_index(drop=True)

    for horizon_min in HORIZONS_MINUTES:
        steps = int(horizon_min * 60 / RESAMPLING_SECONDS)
        for col_src, col_tgt in [
            ("lat",          f"lat_future_{horizon_min}min"),
            ("lon",          f"lon_future_{horizon_min}min"),
            ("altitude",     f"altitude_future_{horizon_min}min"),
            ("flight_phase", f"intent_future_{horizon_min}min"),
        ]:
            df[col_tgt] = df.groupby("trajectory_id")[col_src].shift(-steps)

    log(f"  Targets créées pour horizons {HORIZONS_MINUTES} min")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2-lat1)/2)**2
         + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def evaluate_regression_model(model_name, dataset_name, y_true, y_pred,
                               training_time_sec=None):
    """
    Évalue un modèle de régression.
    y_true et y_pred : DataFrames avec colonnes lat_future, lon_future, altitude_future.
    Retourne un dict de métriques.
    """
    y_true = pd.DataFrame(y_true, columns=["lat_future","lon_future","altitude_future"]) \
        if not isinstance(y_true, pd.DataFrame) else y_true.reset_index(drop=True)
    y_pred = pd.DataFrame(y_pred, columns=["lat_future","lon_future","altitude_future"]) \
        if not isinstance(y_pred, pd.DataFrame) else y_pred.reset_index(drop=True)

    results = {"model": model_name, "dataset": dataset_name}

    for col in ["lat_future", "lon_future", "altitude_future"]:
        results[f"RMSE_{col}"] = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
        results[f"MAE_{col}"]  = mean_absolute_error(y_true[col], y_pred[col])

    dist = haversine_km(y_true["lat_future"], y_true["lon_future"],
                        y_pred["lat_future"], y_pred["lon_future"])
    results["mean_deviation_km"]   = float(np.mean(dist))
    results["median_deviation_km"] = float(np.median(dist))
    results["p95_deviation_km"]    = float(np.percentile(dist, 95))
    results["p99_deviation_km"]    = float(np.percentile(dist, 99))
    results["max_deviation_km"]    = float(np.max(dist))

    if training_time_sec is not None:
        results["training_time_sec"] = training_time_sec

    return results


def reconstruct_future_position(X_original: pd.DataFrame,
                                 y_pred_delta: np.ndarray) -> pd.DataFrame:
    """Reconstruit lat/lon/altitude_future à partir des deltas prédits."""
    return pd.DataFrame({
        "lat_future":      X_original["lat"].values      + y_pred_delta[:, 0],
        "lon_future":      X_original["lon"].values      + y_pred_delta[:, 1],
        "altitude_future": X_original["altitude"].values + y_pred_delta[:, 2],
    })


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES ML (alignées notebook 03)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS_ML = [
    # Position et mouvement de base
    "lat", "lon", "altitude", "velocity", "heading", "vertrate",
    # Cinématique
    "ground_speed_ms", "vertical_rate_fpm", "acceleration", "turn_rate",
    # Direction
    "vx_ms", "vy_ms", "vz_ms", "heading_sin", "heading_cos",
    # Géométrique
    "distance_step_m", "distance_traveled_m", "trajectory_angle",
    "distance_destination_m", "elapsed_time_sec",
    # Rolling
    "roll_mean_velocity",      "roll_std_velocity",
    "roll_mean_altitude",      "roll_std_altitude",
    "roll_mean_vertrate",      "roll_std_vertrate",
    "roll_mean_acceleration",  "roll_std_acceleration",
    "roll_mean_turn_rate",     "roll_std_turn_rate",
    "roll_mean_distance_step_m", "roll_std_distance_step_m",
]


# =============================================================================
# PARTIE ML — multi-horizons (notebook 03)
# =============================================================================

def _prepare_horizon(df_all: pd.DataFrame, horizon_min: int,
                     n_traj: int) -> tuple:
    """
    Prépare X / y_reg / y_clf pour un horizon donné.
    Retourne (X_train, X_val, X_test,
               y_train_reg, y_val_reg, y_test_reg,
               y_train_clf, y_val_clf, y_test_clf)
    """
    # ── Sous-échantillon de trajectoires ─────────────────────────────────────
    all_ids = df_all["trajectory_id"].unique()
    rng = np.random.default_rng(SEED)
    sampled = set(
        rng.choice(all_ids, size=min(n_traj, len(all_ids)), replace=False).astype(str)
    )
    df = df_all[df_all["trajectory_id"].astype(str).isin(sampled)].copy()

    # ── Colonnes génériques pour cet horizon ─────────────────────────────────
    df["lat_future"]      = df[f"lat_future_{horizon_min}min"]
    df["lon_future"]      = df[f"lon_future_{horizon_min}min"]
    df["altitude_future"] = df[f"altitude_future_{horizon_min}min"]
    df["intent_future"]   = df[f"intent_future_{horizon_min}min"]

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    feat_cols  = [c for c in FEATURE_COLS_ML if c in df.columns]
    targets_reg = ["lat_future", "lon_future", "altitude_future"]
    target_clf  = "intent_future"
    required    = feat_cols + targets_reg + [target_clf, "trajectory_id"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    # ── Split par trajectory_id (70 / 15 / 15) ───────────────────────────────
    unique_traj = df["trajectory_id"].unique()
    train_traj, temp_traj = train_test_split(unique_traj, test_size=0.30, random_state=SEED)
    val_traj,   test_traj = train_test_split(temp_traj,   test_size=0.50, random_state=SEED)

    train_mask = df["trajectory_id"].isin(train_traj)
    val_mask   = df["trajectory_id"].isin(val_traj)
    test_mask  = df["trajectory_id"].isin(test_traj)

    X_train = df.loc[train_mask, feat_cols].copy()
    X_val   = df.loc[val_mask,   feat_cols].copy()
    X_test  = df.loc[test_mask,  feat_cols].copy()

    y_train_reg = df.loc[train_mask, targets_reg].copy()
    y_val_reg   = df.loc[val_mask,   targets_reg].copy()
    y_test_reg  = df.loc[test_mask,  targets_reg].copy()

    y_train_clf = df.loc[train_mask, target_clf].copy()
    y_val_clf   = df.loc[val_mask,   target_clf].copy()
    y_test_clf  = df.loc[test_mask,  target_clf].copy()

    log(f"  Horizon {horizon_min}min — Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")
    return (X_train, X_val, X_test,
            y_train_reg, y_val_reg, y_test_reg,
            y_train_clf, y_val_clf, y_test_clf)


def run_ml_horizon(horizon_min: int, df_all: pd.DataFrame,
                   output_dir: Path, n_traj: int = 2000,
                   n_optuna_trials: int = 10) -> dict:
    """
    Pipeline ML complet pour un horizon donné (notebook 03) :
    1. Baseline naïve
    2. Decision Tree (delta)
    3. Random Forest (delta)
    4. Gradient Boosting (delta)
    5. Optuna tuning Random Forest
    6. SVM RBF classification d'intention
    7. Analyse des résidus + figures
    """
    log(f"── ML horizon {horizon_min} min ──────────────────────────────")

    (X_train, X_val, X_test,
     y_train_reg, y_val_reg, y_test_reg,
     y_train_clf, y_val_clf, y_test_clf) = _prepare_horizon(df_all, horizon_min, n_traj)

    # ── Normalisation (fit sur train uniquement) ──────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    h = horizon_min
    joblib.dump(scaler, output_dir / f"ml_scaler_{h}min.pkl")

    # ── Targets delta ─────────────────────────────────────────────────────────
    def make_delta(y_reg, X_raw):
        return pd.DataFrame({
            "delta_lat":      y_reg["lat_future"].values      - X_raw["lat"].values,
            "delta_lon":      y_reg["lon_future"].values      - X_raw["lon"].values,
            "delta_altitude": y_reg["altitude_future"].values - X_raw["altitude"].values,
        })

    y_train_delta = make_delta(y_train_reg, X_train)
    y_val_delta   = make_delta(y_val_reg,   X_val)
    y_test_delta  = make_delta(y_test_reg,  X_test)

    all_results = []

    # ── 1. Baseline naïve ────────────────────────────────────────────────────
    y_pred_naive = X_test[["lat","lon","altitude"]].copy()
    y_pred_naive.columns = ["lat_future","lon_future","altitude_future"]
    all_results.append(
        evaluate_regression_model("Baseline naïve", "test",
                                  y_test_reg, y_pred_naive)
    )

    # ── 2. Decision Tree ─────────────────────────────────────────────────────
    log("  Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=15, min_samples_leaf=20, random_state=SEED)
    t0 = time.time()
    dt.fit(X_train_sc, y_train_delta)
    t_dt = time.time() - t0

    y_pred_dt = reconstruct_future_position(X_test, dt.predict(X_test_sc))
    all_results.append(
        evaluate_regression_model("Decision Tree", "validation",
                                  y_val_reg,
                                  reconstruct_future_position(X_val, dt.predict(X_val_sc)),
                                  t_dt)
    )
    all_results.append(
        evaluate_regression_model("Decision Tree", "test",
                                  y_test_reg, y_pred_dt, t_dt)
    )

    # ── 3. Random Forest ─────────────────────────────────────────────────────
    log("  Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=60, max_depth=18, min_samples_leaf=10,
        max_features="sqrt", n_jobs=-1, random_state=SEED
    )
    t0 = time.time()
    rf.fit(X_train_sc, y_train_delta)
    t_rf = time.time() - t0

    y_pred_rf = reconstruct_future_position(X_test, rf.predict(X_test_sc))
    all_results.append(
        evaluate_regression_model("Random Forest", "validation",
                                  y_val_reg,
                                  reconstruct_future_position(X_val, rf.predict(X_val_sc)),
                                  t_rf)
    )
    all_results.append(
        evaluate_regression_model("Random Forest", "test",
                                  y_test_reg, y_pred_rf, t_rf)
    )

    # ── 4. Gradient Boosting ─────────────────────────────────────────────────
    log("  Gradient Boosting...")
    gb = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, subsample=0.8, random_state=SEED
        ),
        n_jobs=-1
    )
    t0 = time.time()
    gb.fit(X_train_sc, y_train_delta)
    t_gb = time.time() - t0

    y_pred_gb = reconstruct_future_position(X_test, gb.predict(X_test_sc))
    all_results.append(
        evaluate_regression_model("Gradient Boosting", "validation",
                                  y_val_reg,
                                  reconstruct_future_position(X_val, gb.predict(X_val_sc)),
                                  t_gb)
    )
    all_results.append(
        evaluate_regression_model("Gradient Boosting", "test",
                                  y_test_reg, y_pred_gb, t_gb)
    )

    # ── 5. Optuna — Random Forest optimisé ───────────────────────────────────
    log(f"  Optuna RF ({n_optuna_trials} trials)...")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Sous-échantillon pour accélérer le tuning
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(X_train_sc.shape[0], size=min(50000, X_train_sc.shape[0]), replace=False)
    vl_idx = rng.choice(X_val_sc.shape[0],   size=min(20000, X_val_sc.shape[0]),   replace=False)
    X_tr_o  = X_train_sc[tr_idx];  y_tr_o = y_train_delta.iloc[tr_idx].reset_index(drop=True)
    X_vl_o  = X_val_sc[vl_idx];    X_vl_raw_o = X_val.iloc[vl_idx].reset_index(drop=True)
    y_vl_reg_o = y_val_reg.iloc[vl_idx].reset_index(drop=True)

    def objective_rf(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators",    40, 120),
            "max_depth":       trial.suggest_int("max_depth",        8,  25),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 5,  30),
            "min_samples_split":trial.suggest_int("min_samples_split",2, 20),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt","log2",None]),
        }
        m = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
        m.fit(X_tr_o, y_tr_o)
        pred = reconstruct_future_position(X_vl_raw_o, m.predict(X_vl_o))
        dist = haversine_km(y_vl_reg_o["lat_future"], y_vl_reg_o["lon_future"],
                            pred["lat_future"], pred["lon_future"])
        return float(np.mean(dist))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_rf, n_trials=n_optuna_trials)
    log(f"  Meilleur RF Optuna — val dist moy = {study.best_value:.3f} km")
    log(f"  Params : {study.best_params}")

    best_rf = RandomForestRegressor(**study.best_params, random_state=SEED, n_jobs=-1)
    t0 = time.time()
    best_rf.fit(X_train_sc, y_train_delta)
    t_best_rf = time.time() - t0

    y_pred_best_rf = reconstruct_future_position(X_test, best_rf.predict(X_test_sc))
    all_results.append(
        evaluate_regression_model("RF Optuna", "validation",
                                  y_val_reg,
                                  reconstruct_future_position(X_val, best_rf.predict(X_val_sc)),
                                  t_best_rf)
    )
    all_results.append(
        evaluate_regression_model("RF Optuna", "test",
                                  y_test_reg, y_pred_best_rf, t_best_rf)
    )

    # ── Choisir le meilleur modèle de régression ──────────────────────────────
    test_rows = [r for r in all_results if r.get("dataset") == "test"
                 and r["model"] != "Baseline naïve"]
    best_row = min(test_rows, key=lambda r: r["mean_deviation_km"])
    best_model_name = best_row["model"]
    pred_map = {
        "Decision Tree":   y_pred_dt,
        "Random Forest":   y_pred_rf,
        "Gradient Boosting": y_pred_gb,
        "RF Optuna":       y_pred_best_rf,
    }
    best_y_pred = pred_map[best_model_name].reset_index(drop=True)
    log(f"  Meilleur modèle régression : {best_model_name} "
        f"({best_row['mean_deviation_km']:.3f} km)")

    # ── Tableau comparatif régression ─────────────────────────────────────────
    regression_results_df = pd.DataFrame(all_results)
    reg_path = output_dir / f"ml_regression_results_{h}min.csv"
    regression_results_df.to_csv(reg_path, index=False)

    # ── Figure : distance moyenne par modèle ──────────────────────────────────
    test_df = regression_results_df[regression_results_df["dataset"] == "test"]
    plt.figure(figsize=(10, 4))
    plt.bar(test_df["model"], test_df["mean_deviation_km"])
    plt.title(f"Comparaison modèles — Distance moyenne déviation (horizon {h} min)")
    plt.ylabel("km"); plt.xlabel("Modèle")
    plt.xticks(rotation=25, ha="right"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"ml_regression_comparison_{h}min.png", dpi=150)
    plt.close()

    # ── Analyse des résidus ───────────────────────────────────────────────────
    residuals = y_test_reg.reset_index(drop=True) - best_y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["lat_future","lon_future","altitude_future"]):
        ax.hist(residuals[col], bins=50, color="#2563EB")
        ax.set_title(f"Résidus — {col}")
        ax.set_xlabel("Résidu"); ax.set_ylabel("Fréquence")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Résidus — {best_model_name} (horizon {h} min)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"ml_residuals_{h}min.png", dpi=150)
    plt.close()

    # ── SVM RBF — Classification d'intention ─────────────────────────────────
    log("  SVM RBF (classification d'intention)...")

    # PCA uniquement pour SVM
    pca_svm = PCA(n_components=0.95, random_state=SEED)
    X_train_pca = pca_svm.fit_transform(X_train_sc)
    X_val_pca   = pca_svm.transform(X_val_sc)
    X_test_pca  = pca_svm.transform(X_test_sc)

    # LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_clf.astype(str))
    y_val_enc   = le.transform(y_val_clf.astype(str))
    y_test_enc  = le.transform(y_test_clf.astype(str))

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    t0 = time.time()
    svm.fit(X_train_pca, y_train_enc)
    t_svm = time.time() - t0

    y_test_pred_clf = svm.predict(X_test_pca)
    clf_results = {
        "horizon_min": h,
        "model":       "SVM RBF",
        "dataset":     "test",
        "accuracy":    accuracy_score(y_test_enc, y_test_pred_clf),
        "f1_macro":    f1_score(y_test_enc, y_test_pred_clf, average="macro"),
        "f1_weighted": f1_score(y_test_enc, y_test_pred_clf, average="weighted"),
        "training_time_sec": t_svm,
    }
    clf_results_df = pd.DataFrame([clf_results])
    clf_results_df.to_csv(output_dir / f"ml_classification_results_{h}min.csv", index=False)

    # Rapport par classe
    clf_report = classification_report(y_test_enc, y_test_pred_clf,
                                       target_names=le.classes_, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    clf_report_df.to_csv(output_dir / f"ml_classification_report_{h}min.csv")

    # Matrice de confusion normalisée
    cm = confusion_matrix(y_test_enc, y_test_pred_clf)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, aspect="auto", cmap="Blues")
    plt.xticks(range(len(le.classes_)), le.classes_, rotation=30)
    plt.yticks(range(len(le.classes_)), le.classes_)
    plt.xlabel("Classe prédite"); plt.ylabel("Classe réelle")
    plt.colorbar()
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center")
    plt.title(f"Confusion matrix normalisée — SVM RBF (horizon {h} min)")
    plt.tight_layout()
    plt.savefig(output_dir / f"ml_confusion_matrix_{h}min.png", dpi=150)
    plt.close()

    log(f"  SVM RBF — Accuracy={clf_results['accuracy']:.3f}  F1_macro={clf_results['f1_macro']:.3f}")

    # ── Sauvegarde modèles ────────────────────────────────────────────────────
    joblib.dump(best_rf,  output_dir / f"ml_rf_optuna_{h}min.pkl")
    joblib.dump(gb,       output_dir / f"ml_gb_{h}min.pkl")
    joblib.dump(svm,      output_dir / f"ml_svm_{h}min.pkl")
    joblib.dump(pca_svm,  output_dir / f"ml_pca_svm_{h}min.pkl")
    joblib.dump(le,       output_dir / f"ml_label_encoder_{h}min.pkl")

    return {
        "horizon_min":            h,
        "regression_results":     regression_results_df,
        "classification_results": clf_results_df,
        "classification_report":  clf_report_df,
        "confusion_matrix":       cm,
        "best_model_name":        best_model_name,
    }


def run_ml(df_all: pd.DataFrame, output_dir: Path,
           horizons: list, n_traj: int = 2000,
           n_optuna_trials: int = 10) -> list:
    """Lance run_ml_horizon pour chaque horizon, puis génère les graphes comparatifs."""
    multi_outputs = []
    for h in horizons:
        out = run_ml_horizon(h, df_all, output_dir, n_traj, n_optuna_trials)
        multi_outputs.append(out)

    # ── Comparaison multi-horizons ────────────────────────────────────────────
    all_reg = pd.concat([o["regression_results"] for o in multi_outputs], ignore_index=True)
    all_reg["horizon_min"] = all_reg.get("horizon_min", np.nan)

    # Reconstruire la colonne horizon_min depuis les fichiers si absente
    # (chaque df ne contient qu'un horizon — on l'ajoute ici)
    for i, out in enumerate(multi_outputs):
        mask = all_reg.index.isin(out["regression_results"].index +
                                  sum(len(o["regression_results"]) for o in multi_outputs[:i]))
        all_reg.loc[mask, "horizon_min"] = out["horizon_min"]

    # Méthode robuste : reconstruire depuis les outputs
    parts = []
    for out in multi_outputs:
        df_part = out["regression_results"].copy()
        df_part["horizon_min"] = out["horizon_min"]
        parts.append(df_part)
    all_reg = pd.concat(parts, ignore_index=True)
    all_reg.to_csv(output_dir / "ml_multi_horizon_regression.csv", index=False)

    test_reg = all_reg[all_reg["dataset"] == "test"].copy()

    # Figure : évolution de la distance moyenne par modèle et horizon
    plt.figure(figsize=(12, 5))
    for model in test_reg["model"].unique():
        sub = test_reg[test_reg["model"] == model].sort_values("horizon_min")
        plt.plot(sub["horizon_min"], sub["mean_deviation_km"],
                 marker="o", linewidth=2, label=model)
    plt.title("Distance moyenne de déviation selon l'horizon")
    plt.xlabel("Horizon (min)"); plt.ylabel("km")
    plt.xticks(horizons); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ml_multi_horizon_deviation.png", dpi=150)
    plt.close()

    log("  ML multi-horizons — figures et CSV sauvegardés")
    return multi_outputs


# =============================================================================
# PARTIE DL — inchangée par rapport au pipeline original (notebook 04)
# =============================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def make_loaders(X_tr, y_tr, X_vl, y_vl, X_te, y_te, batch_size=128):
    kw = dict(num_workers=0, pin_memory=False)
    tr = DataLoader(TrajectoryDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,  **kw)
    vl = DataLoader(TrajectoryDataset(X_vl, y_vl), batch_size=batch_size, shuffle=False, **kw)
    te = DataLoader(TrajectoryDataset(X_te, y_te), batch_size=batch_size, shuffle=False, **kw)
    return tr, vl, te


class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.25, output_size=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size//2, output_size)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.25, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size//2, output_size)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch,  out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x if self.skip is None else self.skip(x)
        return self.relu(self.net(x) + res)

class TCNPredictor(nn.Module):
    def __init__(self, input_size, channels=None, kernel_size=3, dropout=0.2, output_size=3):
        super().__init__()
        if channels is None: channels = [32, 64, 64]
        layers, in_ch = [], input_size
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(in_ch//2, output_size)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return self.fc(x[:, :, -1])


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5, path="best.pt"):
        self.patience=patience; self.min_delta=min_delta; self.path=path
        self.counter=0; self.best_loss=np.inf; self.stop=False
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss=val_loss; self.counter=0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop=True
    def load_best(self, model):
        model.load_state_dict(torch.load(self.path, map_location=DEVICE))
        return model


def train_model_dl(model, train_loader, val_loader,
                   n_epochs=80, lr=1e-3, patience=12, save_path="best.pt"):
    model     = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    stopper   = EarlyStopping(patience=patience, path=save_path)
    history   = {"train_loss": [], "val_loss": [], "lr": []}
    t0        = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl_losses.append(criterion(model(xb), yb).item())

        t_loss = float(np.mean(tr_losses))
        v_loss = float(np.mean(vl_losses))
        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["lr"].append(cur_lr)

        scheduler.step(v_loss)
        stopper(v_loss, model)

        if epoch % 10 == 0 or epoch == 1 or stopper.stop:
            print(f"    Epoch {epoch:>4}  train={t_loss:.5f}  val={v_loss:.5f}  "
                  f"lr={cur_lr:.2e}  t={time.time()-t0:.0f}s")
        if stopper.stop:
            print(f"    EarlyStopping — epoch {epoch}")
            break

    return stopper.load_best(model), history


FEATURE_COLS_DL = [
    "lat","lon","altitude","velocity","heading","vertrate",
    "ground_speed_ms","acceleration","turn_rate",
    "vx_ms","vy_ms","vz_ms","heading_sin","heading_cos",
    "distance_step_m","roll_mean_velocity","roll_std_velocity",
    "roll_mean_acceleration","roll_std_acceleration",
    "roll_mean_turn_rate","roll_std_turn_rate",
    "flight_phase_enc"
]
SEQ_LEN     = 30
MAX_HORIZON = 30
STEP        = 5


def build_sequences(df, feature_cols, seq_len, max_horizon, step=1):
    X_list, y_list, anchor_list, id_list = [], [], [], []
    for traj_id, group in df.groupby("trajectory_id", sort=False):
        group    = group.reset_index(drop=True)
        feat_arr = group[feature_cols].values.astype(np.float32)
        lat_col  = group["lat"].values.astype(np.float32)
        lon_col  = group["lon"].values.astype(np.float32)
        alt_col  = group["altitude"].values.astype(np.float32)
        n        = len(group)
        for start in range(0, n - seq_len - max_horizon + 1, step):
            end    = start + seq_len
            future = end + max_horizon - 1
            anchor_lat = lat_col[end - 1]
            anchor_lon = lon_col[end - 1]
            anchor_alt = alt_col[end - 1]
            X_list.append(feat_arr[start:end])
            y_list.append([lat_col[future]-anchor_lat,
                           lon_col[future]-anchor_lon,
                           alt_col[future]-anchor_alt])
            anchor_list.append([anchor_lat, anchor_lon, anchor_alt])
            id_list.append(traj_id)

    if not X_list:
        raise ValueError("Aucune séquence générée.")
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(anchor_list, dtype=np.float32),
            id_list)


def run_dl(df_features: pd.DataFrame, output_dir: Path):
    log("-- DL : Préparation des séquences --")

    feat_cols  = [c for c in FEATURE_COLS_DL if c in df_features.columns]
    n_features = len(feat_cols)

    df_features[feat_cols] = df_features[feat_cols].ffill().fillna(0)

    all_ids = df_features["trajectory_id"].unique()
    rng     = np.random.default_rng(SEED)
    sampled = set(rng.choice(all_ids, size=min(800, len(all_ids)), replace=False).astype(str))
    df_s    = df_features[df_features["trajectory_id"].astype(str).isin(sampled)].copy()

    sizes = df_s.groupby("trajectory_id").size()
    valid = sizes[sizes >= 60].index
    df_s  = df_s[df_s["trajectory_id"].isin(valid)].copy()
    log(f"  {df_s['trajectory_id'].nunique()} trajectoires valides")

    X_all, y_all, anchors_all, traj_ids_all = build_sequences(
        df_s, feat_cols, SEQ_LEN, MAX_HORIZON, step=STEP
    )
    log(f"  X={X_all.shape}  y={y_all.shape}")

    unique_traj = list(set(traj_ids_all))
    rng2 = np.random.default_rng(SEED); rng2.shuffle(unique_traj)
    n = len(unique_traj)
    train_ids = set(unique_traj[:int(0.70*n)])
    val_ids   = set(unique_traj[int(0.70*n):int(0.85*n)])
    test_ids  = set(unique_traj[int(0.85*n):])

    arr   = np.array(traj_ids_all)
    mk_tr = np.array([t in train_ids for t in arr])
    mk_vl = np.array([t in val_ids   for t in arr])
    mk_te = np.array([t in test_ids  for t in arr])

    X_train, y_train = X_all[mk_tr], y_all[mk_tr]
    X_val,   y_val   = X_all[mk_vl], y_all[mk_vl]
    X_test,  y_test  = X_all[mk_te], y_all[mk_te]
    anchors_test      = anchors_all[mk_te]
    log(f"  Train:{X_train.shape[0]:,}  Val:{X_val.shape[0]:,}  Test:{X_test.shape[0]:,}")

    nt, t, f = X_train.shape
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train.reshape(-1, f)).reshape(nt, t, f)
    X_val_s   = scaler_X.transform(X_val.reshape(-1, f)).reshape(X_val.shape[0], t, f)
    X_test_s  = scaler_X.transform(X_test.reshape(-1, f)).reshape(X_test.shape[0], t, f)

    scaler_y  = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s   = scaler_y.transform(y_val)
    y_test_s  = scaler_y.transform(y_test)

    train_loader, val_loader, test_loader = make_loaders(
        X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, batch_size=128
    )

    histories = {}
    configs = [
        ("GRU",  GRUPredictor(input_size=n_features)),
        ("LSTM", LSTMPredictor(input_size=n_features)),
        ("TCN",  TCNPredictor(input_size=n_features)),
    ]

    for name, model in configs:
        log(f"  Entraînement {name}...")
        save_path = str(output_dir / f"dl_{name.lower()}_best.pt")
        model, history = train_model_dl(
            model, train_loader, val_loader,
            n_epochs=80, lr=1e-3, patience=12, save_path=save_path
        )
        histories[name] = history

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
        y_pred_orig = scaler_y.inverse_transform(np.vstack(preds))

        lat_pred = anchors_test[:,0] + y_pred_orig[:,0]
        lon_pred = anchors_test[:,1] + y_pred_orig[:,1]
        lat_true = anchors_test[:,0] + y_test[:,0]
        lon_true = anchors_test[:,1] + y_test[:,1]
        dist     = haversine_km(lat_true, lon_true, lat_pred, lon_pred)
        print(f"  {name} — Haversine moy={dist.mean():.3f}km  med={np.median(dist):.3f}km")

    # Courbes d'apprentissage
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, _) in zip(axes, configs):
        h = histories[name]
        ax.plot(h["train_loss"], label="Train")
        ax.plot(h["val_loss"],   label="Val")
        ax.set_title(f"Loss — {name}"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend()
    plt.suptitle("Courbes d'apprentissage — Train vs Validation", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "dl_learning_curves.png", dpi=150)
    plt.close()

    joblib.dump(scaler_X, output_dir / "dl_scaler_X.pkl")
    joblib.dump(scaler_y, output_dir / "dl_scaler_y.pkl")

    log("  DL done — modèles + courbes sauvegardés")
    return histories


# =============================================================================
# MAIN
# =============================================================================
def main():
    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = [args.horizon] if args.horizon else HORIZONS_MINUTES

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   PIPELINE ADS-B — ML + DL — Prédiction de Trajectoires     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Mode      : {args.mode.upper()}")
    print(f"  Horizons  : {horizons} minutes")
    print(f"  Données   : {data_dir}")
    print(f"  Sorties   : {output_dir}")
    print(f"  Device    : {DEVICE}")
    print()

    t_total = time.time()

    # ── Étapes communes ───────────────────────────────────────────────────────
    log("ÉTAPE 1 — Chargement des CSV bruts...")
    df_raw = load_raw_data(data_dir)
    log(f"  {len(df_raw):,} lignes chargées")

    log("ÉTAPE 2 — Nettoyage...")
    df_clean = clean_data(df_raw)
    del df_raw; gc.collect()

    log("ÉTAPE 3 — Resampling 10s + Feature Engineering (colonnes notebook 03)...")
    df_features = resample_and_features(df_clean)
    del df_clean; gc.collect()

    log("ÉTAPE 4 — Création targets multi-horizons (2, 5, 10 min)...")
    df_all = create_all_targets(df_features)

    # ── ML ────────────────────────────────────────────────────────────────────
    if args.mode in ("ml", "both"):
        print()
        log("═══ PARTIE ML (multi-horizons) ═══")
        run_ml(df_all, output_dir,
               horizons=horizons,
               n_traj=args.n_traj,
               n_optuna_trials=args.optuna_trials)

    # ── DL ────────────────────────────────────────────────────────────────────
    if args.mode in ("dl", "both"):
        print()
        log("═══ PARTIE DL ═══")
        run_dl(df_features, output_dir)

    print()
    print(f"  Pipeline terminé en {time.time()-t_total:.1f}s")
    print(f"  Résultats dans : {output_dir}/")
    print()


if __name__ == "__main__":
    main()
