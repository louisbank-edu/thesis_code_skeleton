import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from config import RESULTS_DIR, MODELS_DIR, FEATURE_SETS_OCC, FEATURE_SETS_QTY
from pipeline import load_data, load_grouping
from features import assemble_features

# ------------------------
# Grid-Search Parameter
# ------------------------
TEST_HORIZON = 6
HORIZON_DIR = f"horizon_{TEST_HORIZON}"
RESULTS_DIR = RESULTS_DIR / HORIZON_DIR
MODELS_DIR  = MODELS_DIR / HORIZON_DIR
K_LIST       = [5, 10, 20, 21, 22, 23, 24, 25]     #ensemblegrößen zum testen
ALPHAS       = [0.0, 0.5, 1.0, 2.0]                #gewichtungs parameter α
STRATEGIES   = ["best", "div", "pdiv"]            #auswahlstrategie
MAE_DELTA    = 0.05                               # 5% toleranz für pdiv auswahlstrat

def train_test_split_by_date(df, date_col="date", test_months=TEST_HORIZON):
    cutoff = df[date_col].max() - pd.DateOffset(months=test_months)
    return df[df[date_col] < cutoff], df[df[date_col] >= cutoff]

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    _, test_raw = train_test_split_by_date(df)
    test_raw = test_raw.groupby("Material", sort=False).tail(TEST_HORIZON).reset_index(drop=True)
    y_test = test_raw["demand"].values

    #metriken laden + model_name generieren
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    metrics["model_name"] = metrics.apply(
        lambda r: f"{r['feat_occ']}__{r['feat_qty']}__{r['model']}__{Path(r['grouping']).stem}",
        axis=1
    )

    #diversity laden
    divers = pd.read_csv(RESULTS_DIR / "avg_abs_correlation_per_model.csv")
    if "model_name" not in divers.columns:
        divers = divers.rename(columns={
            divers.columns[0]: "model_name",
            divers.columns[1]: "avg_abs_corr"
        })
    divers["model_name"] = divers["model_name"].str.replace(r"^resid_h\d+_", "", regex=True)
    divers = divers.groupby("model_name", as_index=False)["avg_abs_corr"].mean()

    info = metrics.set_index("model_name")[["feat_occ", "feat_qty", "model", "grouping", "MAE"]]

    preds = pd.DataFrame({
        "Material": test_raw["Material"],
        "date":     test_raw["date"]
    })
    for name, row in info.iterrows():
        feat_occ = row["feat_occ"]
        feat_qty = row["feat_qty"]
        grouping = row["grouping"]
        stem     = Path(grouping).stem

        # modelle laden
        occ = lgb.Booster(model_file=str(MODELS_DIR / f"occ__{feat_occ}__{row['model']}__{stem}.txt"))
        qty = lgb.Booster(model_file=str(MODELS_DIR / f"qty__{feat_qty}__{row['model']}__{stem}.txt"))

        # features bauen
        grp = load_grouping(grouping)
        Xo  = assemble_features(test_raw, FEATURE_SETS_OCC[feat_occ], grp)
        Xq  = assemble_features(test_raw, FEATURE_SETS_QTY[feat_qty], grp)
        # forwardfill PMI
        if "pmi_index" in Xo: Xo["pmi_index"] = Xo["pmi_index"].ffill()
        if "pmi_index" in Xq: Xq["pmi_index"] = Xq["pmi_index"].ffill()

        # vorhersagen
        proba    = occ.predict(Xo)
        qty_pred = qty.predict(Xq)
        preds[name] = proba * qty_pred

    actuals = test_raw.groupby("Material", sort=False)["demand"].apply(list).to_dict()
    ensemble_forecasts = {
        mat: {f"actual_{i}": actuals[mat][i] for i in range(TEST_HORIZON)}
        for mat in actuals
    }
    material_indices = preds.groupby("Material", sort=False).indices

    results = []
    model_list  = list(info.index)
    best_mae    = metrics["MAE"].min()
    perf_thresh = best_mae * (1 + MAE_DELTA)

    for strat in STRATEGIES:
        for K in K_LIST + [len(model_list)]:
            if K > len(model_list):
                continue

            # auswahlstrateige
            if strat == "best":
                subset = metrics.nsmallest(K, "MAE")["model_name"].tolist()
            elif strat == "div":
                subset = divers.nsmallest(K, "avg_abs_corr")["model_name"].tolist()
            else:  # pdiv
                good   = metrics[metrics["MAE"] <= perf_thresh]
                tmp    = good.merge(divers, on="model_name")
                subset = tmp.nsmallest(K, "avg_abs_corr")["model_name"].tolist() or divers.nsmallest(K, "avg_abs_corr")["model_name"].tolist()
            subset = subset[:K]

            if not subset:
                continue

            maes = info.loc[subset, "MAE"].values
            for alpha in ALPHAS:
                weights  = np.exp(-alpha * maes)
                weights /= weights.sum()
                ense_pred = (preds[subset].values * weights).sum(axis=1)
                mad       = np.mean(np.abs(y_test - ense_pred))

                results.append({
                    "strategy": strat,
                    "K": len(subset),
                    "alpha": alpha,
                    "MAD": mad,
                    "models_used": subset
                })

                col = f"{strat}_K{len(subset)}_alpha{alpha}"
                for mat, idxs in material_indices.items():
                    preds_mat = ense_pred[idxs]
                    for i, val in enumerate(preds_mat):
                        ensemble_forecasts[mat][f"{col}_{i}"] = val

    pd.DataFrame(results).to_csv(RESULTS_DIR / "ensemble_comparison.csv", index=False)
    print("ensemble_comparison.csv geschrieben")

    pd.DataFrame(
        [ {"Material": m, **vals} for m, vals in ensemble_forecasts.items() ]
    ).to_csv(RESULTS_DIR / "all_ensemble_forecasts.csv", index=False)
    print("all_ensemble_forecasts.csv geschrieben")

if __name__ == "__main__":
    main()