import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
from config import DATA_PATH
import numpy as np


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Buch.dat."])
    df.rename(columns={"Buch.dat.":"date","Menge in ErfassME":"demand"},inplace=True)
    df['Werk'] = df['Werk'].astype('category').cat.codes
    return df


def load_grouping(path):
    grp = pd.read_csv(path)
    grp.columns = ["Material","group_id"]
    return grp.set_index("Material")["group_id"]


def train_occurrence_model(X,y_occ,cfg):
    model=lgb.LGBMClassifier(objective=cfg["objective"],**cfg["params"])
    model.fit(X,y_occ)
    return model


def train_quantity_model(X,y_qty,cfg):
    model=lgb.LGBMRegressor(objective=cfg["objective"],**cfg["params"])
    model.fit(X,y_qty)
    return model


def combine_predictions(proba_occ,pred_qty):
    # erstmal nur simpel multiplizieren, ggf. noch "schlauer" machen wenn Zeit ist oder andere optionen mit einbauen
    # -> habe es dabei belassen, nur zu multiplizieren
    return proba_occ*pred_qty


def evaluate(y_true,y_pred,y_occ_true=None,proba_occ=None):
    # absoluten error berechnen um fehler nach specification von SAP zu haben
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    res={"MAE":mean_absolute_error(y_true,y_pred), "MAD": float(errors.mean())}
    if y_occ_true is not None:
        res["AUC"]=roc_auc_score(y_occ_true,proba_occ)
    return res
