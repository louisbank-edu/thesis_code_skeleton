import pandas as pd
import numpy as np
from pathlib import Path

from config import EXPERIMENTS, FEATURE_SETS_OCC, FEATURE_SETS_QTY, MODELS, RESULTS_DIR, MODELS_DIR
from pipeline import (
    load_data,
    load_grouping,
    train_occurrence_model,
    train_quantity_model,
    combine_predictions,
    evaluate
)
from features import assemble_features

FORECAST_HORIZON = 6
HORIZON_ID = f"horizon_{FORECAST_HORIZON}"
RESULTS_DIR = RESULTS_DIR / HORIZON_ID
MODELS_DIR = MODELS_DIR / HORIZON_ID

def train_test_split_by_date(df, date_col="date", test_months=3):
    cutoff = df[date_col].max() - pd.DateOffset(months=test_months)
    return df[df[date_col] < cutoff], df[df[date_col] >= cutoff]

def main():
    # ensure output dirs exist
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # load and split data
    df_full      = load_data()
    train_raw, test_raw = train_test_split_by_date(df_full, test_months=FORECAST_HORIZON)

    # Für Forecast: kombiniere train + test in eine Zeile
    df_forecast = pd.concat([train_raw, test_raw], ignore_index=True)

    # Zum Sammeln aller Residual-DFs
    resid_dfs = []

    # Prepare targets
    y_train     = train_raw["demand"]
    y_occ_train = (y_train > 0).astype(int)
    y_test      = test_raw["demand"]
    y_occ_test  = (y_test > 0).astype(int)

    summary = []

    for exp in EXPERIMENTS:
        print(f"Experiment: {exp}")
        grouping = load_grouping(exp["grouping"])

        # — train occurrence model —
        X_train_occ = assemble_features(
            train_raw,
            FEATURE_SETS_OCC[exp["feat_occ"]],
            grouping
        )
        occ_mod = train_occurrence_model(
            X_train_occ,
            y_occ_train,
            MODELS[exp["model"]]["occurrence"]
        )

        # — train quantity model —
        pos_idx = y_occ_train[y_occ_train == 1].index
        X_train_qty = assemble_features(
            train_raw.loc[pos_idx],
            FEATURE_SETS_QTY[exp["feat_qty"]],
            grouping
        )
        qty_mod = train_quantity_model(
            X_train_qty,
            y_train.loc[pos_idx],
            MODELS[exp["model"]]["quantity"]
        )

        # — iterative Forecast & Residuen sammeln —
        df_fc = df_forecast.copy()
        model_name = (
            f"{exp['feat_occ']}__{exp['feat_qty']}__"
            f"{exp['model']}__{Path(exp['grouping']).stem}"
        )
        dates = sorted(test_raw["date"].unique())

        # Data‐Dict initialisieren
        data = {"Material": []}
        for h in range(1, FORECAST_HORIZON + 1):
            data[f"resid_h{h}_{model_name}"] = []

        for h, dt in enumerate(dates, start=1):
            mask      = df_fc["date"] == dt
            true_vals = df_fc.loc[mask, "demand"].values
            mats      = df_fc.loc[mask, "Material"].values

            # Features bauen auf gesamter Historie inkl. vorheriger Vorhersagen
            X_occ = assemble_features(df_fc, FEATURE_SETS_OCC[exp["feat_occ"]], grouping)
            X_qty = assemble_features(df_fc, FEATURE_SETS_QTY[exp["feat_qty"]], grouping)
            # PMI forward‐fill
            if "pmi_index" in X_occ.columns:
                X_occ["pmi_index"] = X_occ["pmi_index"].ffill()
            if "pmi_index" in X_qty.columns:
                X_qty["pmi_index"] = X_qty["pmi_index"].ffill()

            proba    = occ_mod.predict_proba(X_occ[mask])[:, 1]
            qty_pred = qty_mod.predict(X_qty[mask])
            pred     = proba * qty_pred

            # Für den nächsten Schritt in 'demand' schreiben
            df_fc.loc[mask, "demand"] = pred

            resid = true_vals - pred
            if h == 1:
                data["Material"] = list(mats)
            data[f"resid_h{h}_{model_name}"] = list(resid)

        # Residual‐DF für dieses Experiment
        df_resid_exp = pd.DataFrame(data)
        resid_dfs.append(df_resid_exp)

        # — Evaluate für Summary —
        X_test_occ = assemble_features(test_raw, FEATURE_SETS_OCC[exp["feat_occ"]], grouping)
        X_test_qty = assemble_features(test_raw, FEATURE_SETS_QTY[exp["feat_qty"]], grouping)
        proba_test    = occ_mod.predict_proba(X_test_occ)[:, 1]
        qty_pred_test = qty_mod.predict(X_test_qty)
        final_pred    = combine_predictions(proba_test, qty_pred_test)
        metrics       = evaluate(test_raw["demand"], final_pred, y_occ_test, proba_test)
        summary.append({**exp, **metrics})

        # — Modelle speichern —
        stem = Path(exp["grouping"]).stem
        occ_mod.booster_.save_model(
            MODELS_DIR / f"occ__{exp['feat_occ']}__{exp['model']}__{stem}.txt"
        )
        qty_mod.booster_.save_model(
            MODELS_DIR / f"qty__{exp['feat_qty']}__{exp['model']}__{stem}.txt"
        )

    # — Summary exportieren —
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    # — Alle Residuals zusammenführen und speichern —
    df_all_resid = resid_dfs[0].set_index("Material")
    for dfr in resid_dfs[1:]:
        df_all_resid = df_all_resid.join(dfr.set_index("Material"), how="outer")
    df_all_resid.reset_index().to_csv(
        RESULTS_DIR / "all_models_residuals.csv", index=False
    )

if __name__ == "__main__":
    main()
