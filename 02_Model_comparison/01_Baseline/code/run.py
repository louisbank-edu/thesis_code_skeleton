import os
from function import run_forecasts_with_groupings


grouping_files_list = ["PATH_GROUPING_FILES/grouping_semantic_auto_named.csv",
                       "PATH_GROUPING_FILES/grouping_all.csv",
                      "PATH_GROUPING_FILES/grouping_materialwise.csv",
                        "PATH_GROUPING_FILES/grouping_adi_cv2.csv",
                        "PATH_GROUPING_FILES/grouping_intermittency_similarity.csv",
]

if __name__ == "__main__":
    data_file = "DATA_PATH/monthly_complete.csv"
    grouping_files = grouping_files_list
    # ACHTUNG: IMMER SOWOHL TEST HORIZON ALS AUCH HIER BEIM OUTPUT DIR DAS ÄNDERN
    # (kein gutes coding, aber einfach)
    output_plots_dir = os.path.join("..", "01_plots", "03_monthly/horizon_2")
    agg_freq = "ME"  # monthly frequency
    extra_features = ["Werk", "Kunde", "txn_count", "unique_recipients", "Bewegungsart"]
    test_horizon = 2

    run_forecasts_with_groupings(data_file, grouping_files, output_plots_dir, agg_freq, extra_features, test_horizon)

    """"
    print("2 DONE")
    run_forecasts_with_groupings(data_file, grouping_files, os.path.join("..", "01_plots", "03_monthly/horizon_3"), agg_freq, extra_features, 3)
    print("3 DONE")
    run_forecasts_with_groupings(data_file, grouping_files, os.path.join("..", "01_plots", "03_monthly/horizon_4"), agg_freq, extra_features, 4)
    print("4 DONE")
    run_forecasts_with_groupings(data_file, grouping_files, os.path.join("..", "01_plots", "03_monthly/horizon_5"), agg_freq, extra_features, 5)
    print("5 DONE")
    run_forecasts_with_groupings(data_file, grouping_files, os.path.join("..", "01_plots", "03_monthly/horizon_6"), agg_freq, extra_features, 6)
    print("6 DONE")
    """