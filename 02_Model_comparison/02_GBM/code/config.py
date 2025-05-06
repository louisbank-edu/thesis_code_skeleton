from pathlib import Path
BASE_DIR = Path("BASE_DIR")
DATA_PATH = BASE_DIR / ".../monthly_complete.csv"
GBM_DIR = BASE_DIR / "02_GBM"
GROUPING_DIR = BASE_DIR / "03_grouping"
RESULTS_DIR = GBM_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"


FEATURE_SETS_OCC = {
    "occ_basic": [
        "month", "Werk", "group_id"
    ],
    "occ_seasonal": [
        "month_sin", "month_cos", "quarter",
        "Werk", "group_id"
    ],
    "occ_intervals": [
        "month", "Werk", "group_id",
        "periods_since_last_demand",
        "zero_run_length",
        "avg_interarrival"
    ],
    "occ_exogenous": [
        "month", "Werk", "group_id",
        "periods_since_last_demand", "zero_run_length",
        "holiday_flag",
        "holiday_ratio",
        "pmi_index"
    ],
    "occ_full": [
        "month", "month_sin", "month_cos", "quarter",
        "Werk", "group_id",
        "periods_since_last_demand", "zero_run_length", "avg_interarrival",
        "holiday_flag", "pmi_index",
        "holiday_ratio",
        "demand_lag_1", "demand_lag_3",
        "demand_rolling_count_3", "demand_rolling_count_6"
    ]
}
FEATURE_SETS_QTY = {
    "qty_basic": [
        "month", "Werk", "group_id"
    ],
    "qty_lags": [
        "month", "Werk", "group_id",
        "demand_lag_1", "demand_lag_3",
        "demand_rolling_mean_3"
    ],
    "qty_dist": [
        "month", "Werk", "group_id",
        "last_positive_demand",
        "last_3_positive_mean", "last_3_positive_median",
        "demand_expanding_mean", "demand_expanding_std"
    ],
    "qty_exogenous": [
        "month", "Werk", "group_id",
        "pmi_index",
        "demand_lag_1", "demand_lag_3", "demand_lag_6",
        "demand_rolling_mean_3", "demand_rolling_std_3"
    ],
    "qty_full": [
        "month", "month_sin", "month_cos", "quarter",
        "Werk", "group_id",
        "demand_lag_1", "demand_lag_3", "demand_lag_6", "demand_lag_12",
        "demand_rolling_mean_3", "demand_rolling_std_3",
        "demand_rolling_mean_6", "demand_rolling_std_6",
        "last_positive_demand", "last_3_positive_mean",
        "pmi_index"]}


# ggf. verschiedene Modelle (erstmal nur lightgbm, weil das eh am besten sein soll, auch laut M5 competition)
# -> es wurde nur lightgbm getestet
MODELS = {
    "lightgbm": {
        "occurrence": {
            "framework": "lightgbm",
            "objective": "binary",
            "params": {"boosting_type":"gbdt","class_weight":"balanced","n_estimators":100,"learning_rate":0.1,"min_child_samples":1}
        },
        "quantity": {
            "framework": "lightgbm",
            "objective": "tweedie",
            "params": {"boosting_type":"gbdt","tweedie_variance_power":1.5,"n_estimators":100,"learning_rate":0.1}
        }
    }
}

# alle gruppierungsdateien aus demm ordner ziehen
GROUPINGS = [str(p) for p in GROUPING_DIR.glob("*.csv")]

# daraus jetzt ein "grid" aller Kombinationen machen, die durchprobiert werden sollen
EXPERIMENTS = [
     {
       "feat_occ": feat_occ,
       "feat_qty": feat_qty,
       "model": m,
       "grouping": g
     }
     for feat_occ in FEATURE_SETS_OCC
     for feat_qty in FEATURE_SETS_QTY
     for m in MODELS
     for g in GROUPINGS
 ]