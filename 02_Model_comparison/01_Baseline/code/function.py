import os
import pandas as pd
import numpy as np
from modules import data_processing, forecasting, evaluation, plotting

def run_forecasts_with_groupings(
    data_file,
    grouping_files,
    output_root_dir,
    agg_freq,
    extra_features,
    test_horizon=12
):
    # daten einlesen
    print(f"Lade Daten aus {data_file}")
    data = pd.read_csv(data_file, parse_dates=["Buch.dat."])
    #zur einfacheren Handhabung umbenennen in demand
    data.rename(columns={"Menge in ErfassME": "demand"}, inplace=True)

    #jetzt über die gruppierungsdateien gehen
    for grp_file in grouping_files:
        #gruppierungsnamen ziehen
        grp_name = os.path.splitext(os.path.basename(grp_file))[0]
        grp_dir = os.path.join(output_root_dir, grp_name)
        os.makedirs(grp_dir, exist_ok=True)

        print(f"\n=== Grouping '{grp_name}' ===")
        grp_df = pd.read_csv(grp_file)
        # die grouping csvs sind immer so aufgebaut, dass sie eine spalte "Material" haben und eine spalte für die "GroupID"
        group_ids = grp_df["GroupID"].unique()

        grouping_overall_results = []
        grouping_forecasts = []  # collect forecasts per grouping

        # pro gruppe über alle group ids iterieren.
        for gid in group_ids:
            materials_in_current_group = grp_df.loc[grp_df["GroupID"] == gid, "Material"].tolist()
            out = os.path.join(grp_dir, f"group_{gid}")
            os.makedirs(out, exist_ok=True)

            print(f"\n-- Group {gid}: {len(materials_in_current_group)} materials --")
            #array anlegen, in dem wir die overall results speichern
            overall_results = []
            # alle methoden, die wir uns ansehen bzw miteinander vergleichen möchten
            methods = [
                "superpessimist",
                "croston", "tsb", "sba", "sbj", "ewma",
                "holt", "holt_winters",
                "poisson", *[f"poisson_{k}" for k in ["median","q10","q25","q75","q90"]],
                "nbd", *[f"nbd_{k}" for k in ["median","q10","q25","q75","q90"]],
                "zip", *[f"zip_{k}" for k in ["median","q10","q25","q75","q90"]],
                "nhpp", *[f"nhpp_{k}" for k in ["median","q10","q25","q75","q90"]],
                *[f"willemain_{k}" for k in ["mean","median","q10","q25","q75","q90"]],
                *[f"hybrid_{v}" for v in ["base", "a", "b", "c", "d", "e", "c_tuned", "c_ewma", "c_svr_ewma"]],
                "gbm_direct", "gbm_hybrid"
            ]
            # dict um zu zählen, wie häufig die methoden jeweils den besten mean hatten
            best_mean_counts = dict.fromkeys(methods, 0)
            # und dasselbe für median
            best_median_counts = dict.fromkeys(methods, 0)

            # daten preprocessen und trainingsdaten sammeln
            train_dfs = []
            mat_data = {}
            for material in materials_in_current_group:
                material_dataframe = data_processing.preprocess_material_data(
                    data, material, agg_freq, extra_features=extra_features
                )
                # wenn das material weniger als testhorizont + 12 (also ein jahr vorlauf)
                # datenpunkte hat, skippen.
                # ALLERDINGS TODO: das wird nie passieren, weil ich ja auch die Vorperioden mit Nullen fülle
                # also noch überlegen, ob ich das anders gestalte (dann könnte es halt so extrem viele materialien
                # raushauen)
                if material_dataframe.shape[0] < test_horizon + 12:
                    print(f"   skip {material}: only {material_dataframe.shape[0]} rows")
                    continue
                train_dfs.append(material_dataframe.iloc[:-test_horizon].copy())
                # mat data ordnet jedem material seinen entsprechenden material dataframe zu (also die serie)
                mat_data[material] = material_dataframe

            # das sind die varianten, die wir für die features haben
            variants = ["base", "a", "b", "c", "d", "e", "c_tuned", "c_ewma", "c_svr_ewma"]
            group_hybrids_by_variant_dict = {}
            for variant in variants:
                trained_model = forecasting.train_hybrid_variant_group(
                    train_dfs,
                    variant,
                    extra_features
                )
                group_hybrids_by_variant_dict[variant] = trained_model
            group_gbm_direct = forecasting.train_direct_gbm_base_group(train_dfs, extra_features)
            group_gbm_hybrid = forecasting.train_hybrid_gbm_base_group(train_dfs, extra_features)

            # jetzt über alle materialien iterieren (in der aktuellen Gruppe)
            for material, material_dataframe in mat_data.items():
                print(f"Material {material}")
                train = material_dataframe.iloc[:-test_horizon].copy()
                test = material_dataframe.iloc[-test_horizon:].copy()
                y_true = test["demand"].values
                forecast_horizon_for_test = test_horizon

                # speichere für das material für jede methode den baseline fc und ob es einen fallback gab in nem dict
                baseline_fc, baseline_fb = {}, {}

                fc, fb = forecasting.superpessimist_forecast(train["demand"], forecast_horizon_for_test)
                baseline_fc["superpessimist"] = fc; baseline_fb["superpessimist"] = fb

                for name, fn in [
                    ("croston", forecasting.croston_forecast),
                    ("tsb",     forecasting.tsb_forecast),
                    ("sba",     forecasting.sba_forecast),
                    ("sbj",     forecasting.sbj_forecast),
                    ("ewma",    forecasting.ewma_forecast),
                    ("holt",    forecasting.holt_forecast),
                    ("holt_winters", forecasting.holt_winters_forecast),
                ]:
                    # über die forecasting methoden iterieren, um die baseline forecasts zu bekommen und fb
                    if name in ["holt", "holt_winters"]:
                        args = dict(alpha=0.1, beta=0.1)
                        if name=="holt_winters": args.update(gamma=0.1, seasonal_periods=12, seasonal='add')
                        fc, fb = forecasting.recursive_forecast_base(fn, train["demand"], forecast_horizon_for_test, **args)
                    else:
                        fc, fb = forecasting.recursive_forecast_base(fn, train["demand"], forecast_horizon_for_test, alpha=0.1)
                    baseline_fc[name] = fc; baseline_fb[name] = fb

                # hier vorab definieren, ich möchte ja bei willemain nicht nur "mean" sondern auch die verschiedenen quantiles testen
                willemain_quantiles = {
                    "mean": "mean",
                    "median": "median",
                    "q10": 0.10,
                    "q25": 0.25,
                    "q75": 0.75,
                    "q90": 0.90,
                }
                # und auch für die anderen parametrischen verteilungsmethoden
                quantile_variants = {
                    "q10": 0.10,
                    "q25": 0.25,
                    "median": 0.50,
                    "q75": 0.75,
                    "q90": 0.90,
                }

                for name, fn in [
                    ("poisson", forecasting.poisson_forecast),
                    ("nbd",     forecasting.nbd_forecast),
                    ("zip",     forecasting.zip_forecast),
                    ("nhpp",    forecasting.nhpp_poisson_forecast)
                ]:
                    if name=="nhpp":
                        fc, fb = fn(train["demand"], forecast_horizon_for_test, method='ewma', alpha=0.1, quantile=None)
                    else:
                        fc, fb = fn(train["demand"], forecast_horizon_for_test, quantile=None)
                    baseline_fc[name] = fc; baseline_fb[name] = fb

                    for suffix, q in quantile_variants.items():
                        meth_q = f"{name}_{suffix}"
                        if name=="nhpp":
                            fc, fb = fn(train["demand"], forecast_horizon_for_test, method='ewma', alpha=0.1, quantile=q)
                        else:
                            fc, fb = fn(train["demand"], forecast_horizon_for_test, quantile=q)
                        baseline_fc[meth_q] = fc; baseline_fb[meth_q] = fb

                # now inject one willemain per quantile:
                for suffix, quantile in willemain_quantiles.items():
                    meth_name = f"willemain_{suffix}"
                    fc, fb = forecasting.willemain_boostrap_pointforecast_multistep_mean(
                        train["demand"],
                        forecast_horizon_for_test,
                        n_bootstrap=1000,
                        central_tendency=quantile
                    )
                    baseline_fc[meth_name] = fc; baseline_fb[meth_name] = fb

                #jetzt noch für die hybriden funktionen
                hybrid_fc, hybrid_fb = {}, {}
                for v in variants:
                    key = f"hybrid_{v}"
                    # rufe die funktion für das rekursive mehrschritt verfahren auf
                    fc, fb = forecasting.recursive_rf_hybrid_forecast(
                        group_hybrids_by_variant_dict[v], train, forecast_horizon_for_test
                    )
                    hybrid_fc[key] = fc; hybrid_fb[key] = fb

                # jetzt noch für die separaten GBM
                gbm_direct_fc, gbm_direct_fb = forecasting.recursive_direct_gbm_forecast_pretrained_base(
                    group_gbm_direct, train, forecast_horizon_for_test
                )
                gbm_hybrid_fc, gbm_hybrid_fb = forecasting.recursive_hybrid_gbm_forecast_pretrained_base(
                    group_gbm_hybrid, train, forecast_horizon_for_test
                )

                # mergen und evaluieren
                all_fc = {**baseline_fc, **hybrid_fc, "gbm_direct": gbm_direct_fc, "gbm_hybrid": gbm_hybrid_fc}
                all_fb = {**baseline_fb, **hybrid_fb, "gbm_direct": gbm_direct_fb, "gbm_hybrid": gbm_hybrid_fb}

                summary = evaluation.error_summary(y_true, all_fc)
                nonfb_summary, fb_ratio = evaluation.error_summary_excluding_fallbacks(
                    y_true, all_fc, all_fb
                )

                best_method_for_this_material_by_mean = min(summary, key=lambda k: summary[k]["mean"])
                best_method_for_this_material_by_median = min(summary, key=lambda k: summary[k]["median"])
                best_mean_counts[best_method_for_this_material_by_mean] += 1
                best_median_counts[best_method_for_this_material_by_median] += 1

                # plotting und summary für das spezifische material
                mat_dir = os.path.join(out, f"material_{material}")
                os.makedirs(mat_dir, exist_ok=True)
                plotting.plot_forecast_comparison(material, train, test, all_fc,
                                                  os.path.join(mat_dir, "forecast_comparison.png"))
                # drei besten methoden rausfiltern (abgesehen von superpessimist)
                sorted_by_mean = sorted(
                    summary.items(),
                    key=lambda item: item[1]["mean"]
                )
                top3_methods = [method for method, metrics in sorted_by_mean if method != "superpessimist"][:3]
                # subset aus den drei besten methoden exklusive superpessimist
                subset = {meth: all_fc[meth] for meth in top3_methods} # starke abkürzung für "Methode", wie ich finde
                #plot speichern
                plotting.plot_forecast_comparison(
                    material,
                    train,
                    test,
                    subset,
                    os.path.join(mat_dir, "forecast_comparison_subset.png")
                )

                #zeug an die summary function geben für die evaluation
                txt = evaluation.format_summary_text(
                    material, train.index, test.index,
                    summary, best_method_for_this_material_by_mean, best_method_for_this_material_by_median,
                    nonfb_summary, fb_ratio
                )
                with open(os.path.join(mat_dir, "summary.txt"), "w") as f:
                    f.write(txt)

                # ergebnisse für diese spezifische material sammeln und zum overall df appenden
                row = {"Material": material}
                for meth in summary:
                    row[f"{meth}_mean"]       = summary[meth]["mean"]
                    row[f"{meth}_median"]     = summary[meth]["median"]
                    row[f"{meth}_std"]        = summary[meth]["std"]
                    row[f"{meth}_mean_nofb"]  = nonfb_summary[meth]["mean"]
                    row[f"{meth}_median_nofb"]= nonfb_summary[meth]["median"]
                    row[f"{meth}_fb_ratio"]   = fb_ratio[meth]
                row["best_method_mean"]   = best_method_for_this_material_by_mean
                row["best_method_median"] = best_method_for_this_material_by_median
                row["n_observations"]     = material_dataframe.shape[0]
                overall_results.append(row)

                # collect forecasts for this material
                fc_row = {"Material": material}
                # actual values
                for idx, actual in enumerate(y_true):
                    fc_row[f"actual_{idx}"] = actual
                # forecasted values for each method
                for meth, preds in all_fc.items():
                    for idx, pred in enumerate(preds):
                        fc_row[f"{meth}_{idx}"] = pred
                grouping_forecasts.append(fc_row)

            # die overall results abspeichern
            group_df = pd.DataFrame(overall_results)
            group_df.to_csv(os.path.join(out, "overall_performance.csv"), index=False)

            grp_txt = evaluation.format_overall_summary(
                group_df, best_mean_counts, best_median_counts, methods, include_nofb=True
            )
            with open(os.path.join(out, "00_overall_summary.txt"), "w") as f:
                f.write("\n".join(grp_txt))
            plotting.plot_best_method_histogram(
                best_mean_counts, "Best Methods by Mean MAD",
                os.path.join(out, "best_method_histogram_mean.png"))
            plotting.plot_best_method_histogram(
                best_median_counts, "Best Methods by Median MAD",
                os.path.join(out, "best_method_histogram_median.png"))

            grouping_overall_results.extend(overall_results)

        # und noch die summary über alle gruppen hinweg innerhalb der gruppierung
        all_groups_df = pd.DataFrame(grouping_overall_results)
        all_groups_df.to_csv(os.path.join(grp_dir, "overall_all_groups.csv"), index=False)

        # save forecasts for all materials in this grouping
        forecasts_df = pd.DataFrame(grouping_forecasts)
        forecasts_df.to_csv(os.path.join(grp_dir, "all_forecasts.csv"), index=False)

        grouping_mean_counts = all_groups_df["best_method_mean"].value_counts().to_dict()
        grouping_median_counts = all_groups_df["best_method_median"].value_counts().to_dict()
        all_txt = evaluation.format_overall_summary(
            all_groups_df, grouping_mean_counts, grouping_median_counts, methods, include_nofb=True
        )
        with open(os.path.join(grp_dir, "00_overall_summary_all_groups.txt"), "w") as f:
            f.write("\n".join(all_txt))
        plotting.plot_best_method_histogram(
            grouping_mean_counts, "All Groups: Best by Mean MAD",
            os.path.join(grp_dir, "best_method_histogram_mean_all_groups.png"))
        plotting.plot_best_method_histogram(
            grouping_median_counts, "All Groups: Best by Median MAD",
            os.path.join(grp_dir, "best_method_histogram_median_all_groups.png"))