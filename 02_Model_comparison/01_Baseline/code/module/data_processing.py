import pandas as pd


def preprocess_material_data(data, material_id, freq, extra_features=[]):

    df = data[data["Material"] == material_id].copy()
    if df.empty:
        return pd.DataFrame()

    df.set_index("Buch.dat.", inplace=True)

    agg_dict = {"demand": "sum"}

    for feature in extra_features:
        if feature in df.columns:
            agg_dict[feature] = "first"

    df_agg = df.resample(freq).agg(agg_dict)
    df_agg["demand"] = df_agg["demand"].fillna(0)
    return df_agg
