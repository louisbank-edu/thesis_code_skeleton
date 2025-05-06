import pandas as pd
import numpy as np
import re
from pathlib import Path

# ---------------------------
# Holiday ratio & flag lookup
# ---------------------------
# percentage of days in month that are holidays -> von ChatGPT mit separatem Prompt generiert
_HOLIDAY_PERCENT = {
    1: 3.2,  2: 2.9,  3: 2.9,  4: 3.3,
    5: 3.9,  6: 3.0,  7: 2.6,  8: 2.9,
    9: 2.7, 10: 3.2, 11: 3.0, 12: 3.2
}
# ratio for holiday days
_HOLIDAY_RATIO = {m: p/100.0 for m, p in _HOLIDAY_PERCENT.items()}
# months containing a major national holiday in Germany
_HOLIDAY_FLAG_MONTHS = {1, 5, 10, 12}  # Jan, May, Oct, Dec

# ---------------------------
# PMI exogenous lookup
# ---------------------------
_PMI_PATH = Path("EXOG_DATA/pmi.csv")

_pmi_raw = pd.read_csv(_PMI_PATH, parse_dates=["Date"])
_pmi_raw["Period"] = _pmi_raw["Date"].dt.to_period("M")
_PMI_SERIES = _pmi_raw.set_index("Period")["ActualValue"]

# ---------------------------
# Generator functions
# ---------------------------

def generate_pmi_feature(df):
    periods = df["date"].dt.to_period("M")
    return pd.DataFrame(
        {"pmi_index": periods.map(_PMI_SERIES)},
        index=df.index
    )

def generate_calendar_features(df, feature_list=None):
    out = pd.DataFrame(index=df.index)
    month = df['date'].dt.month
    out['month']   = month
    out['quarter'] = df['date'].dt.quarter

    if feature_list is None:
        return out

    # seasonality encoding
    if 'month_sin' in feature_list:
        out['month_sin'] = np.sin(2 * np.pi * month / 12)
    if 'month_cos' in feature_list:
        out['month_cos'] = np.cos(2 * np.pi * month / 12)

    # holiday ratio
    if 'holiday_ratio' in feature_list:
        out['holiday_ratio'] = month.map(_HOLIDAY_RATIO)
    # holiday flag (1 if month has a major national holiday)
    if 'holiday_flag' in feature_list:
        out['holiday_flag'] = month.map(lambda m: 1 if m in _HOLIDAY_FLAG_MONTHS else 0)

    return out


def generate_group_id(df, grouping):
    mapper = grouping.to_dict()
    series = df['Material'].map(mapper)
    codes = series.astype('category').cat.codes
    return pd.DataFrame({'group_id': codes}, index=df.index)


def generate_lag_features(df, lag_map: dict[str, list[int]]):
    out = pd.DataFrame(index=df.index)
    df_sorted = df.sort_values(['Material', 'date'])
    for col, lags in lag_map.items():
        for lag in lags:
            feat = f"{col}_lag_{lag}"
            out[feat] = (
                df_sorted.groupby('Material')[col]
                         .shift(lag)
                         .fillna(0)
                         .reindex(df.index)
            )
    return out


def generate_rolling_features(df, roll_map: dict[str, dict[str, list[int]]]):
    out = pd.DataFrame(index=df.index)
    df_sorted = df.sort_values(['Material', 'date'])
    for col, stats in roll_map.items():
        series = df_sorted.groupby('Material')[col].shift(1)
        for stat, windows in stats.items():
            for w in windows:
                feat   = f"{col}_rolling_{stat}_{w}"
                rolled = getattr(series.rolling(window=w), stat)()
                out[feat] = rolled.fillna(0).reindex(df.index)
    return out


def generate_expanding_features(df, exp_map: dict[str, list[str]]):
    out = pd.DataFrame(index=df.index)
    df_sorted = df.sort_values(['Material', 'date'])
    for col, stats in exp_map.items():
        series = df_sorted.groupby('Material')[col].shift(1)
        for stat in stats:
            feat = f"{col}_expanding_{stat}"
            agg = series.expanding().mean() if stat=='mean' else series.expanding().std()
            out[feat] = agg.fillna(0).reindex(df.index)
    return out


def generate_interval_features(df):
    out = pd.DataFrame(index=df.index)
    df2 = df.sort_values(['Material','date'])
    psld, zr, ai = [], [], []
    last_pos = None
    run_zero = 0
    inters = []

    for idx, demand in zip(df2.index, df2['demand']):
        # first: record the old-state features
        date_idx = df2.loc[idx,'date'].to_period('M')
        psld.append(np.nan if last_pos is None
                    else (date_idx - df2.loc[last_pos,'date'].to_period('M')).n)
        zr.append(run_zero)
        ai.append(np.nan if not inters else sum(inters)/len(inters))

        # then update your counters _after_ you’ve captured them
        if demand > 0:
            if last_pos is not None:
                months = (date_idx - df2.loc[last_pos,'date'].to_period('M')).n
                inters.append(months)
            last_pos = idx
            run_zero = 0
        else:
            run_zero += 1

    out['periods_since_last_demand'] = pd.Series(psld, index=df2.index).reindex(df.index).fillna(0)
    out['zero_run_length']            = pd.Series(zr, index=df2.index).reindex(df.index)
    out['avg_interarrival']           = pd.Series(ai, index=df2.index).reindex(df.index).fillna(0)
    return out



def generate_rolling_count_features(df, count_map: dict[str,list[int]]):
    out=pd.DataFrame(index=df.index)
    df2=df.sort_values(['Material','date'])
    for col,ws in count_map.items():
        ser=df2.groupby('Material')[col].shift(1).gt(0).astype(int)
        grp=ser.groupby(df2['Material'])
        for w in ws:
            feat=f"{col}_rolling_count_{w}"
            out[feat]=grp.rolling(window=w,min_periods=1).sum().droplevel(0).reindex(df.index).fillna(0)
    return out


def generate_last_positive_features(df,n_list:list[int]):
    out=pd.DataFrame(index=df.index)
    df2=df.sort_values(['Material','date'])
    pos=df2.groupby('Material')['demand'].shift(1).where(df2['demand'].shift(1)>0)
    last=pos.groupby(df2['Material']).ffill().reindex(df.index).fillna(0)
    out['last_positive_demand']=last
    for n in n_list:
        grp=pos.groupby(df2['Material'])
        out[f'last_{n}_positive_mean']=grp.rolling(window=n,min_periods=1).mean().droplevel(0).reindex(df.index).fillna(0)
        out[f'last_{n}_positive_median']=grp.rolling(window=n,min_periods=1).median().droplevel(0).reindex(df.index).fillna(0)
    return out


def generate_interaction_features(X:pd.DataFrame, interactions:list[str]):
    out=pd.DataFrame(index=X.index)
    for feat in interactions:
        l,r=[s.strip() for s in feat.split('×')]
        out[feat]=X[l]*X[r]
    return out


def assemble_features(df, feature_list, grouping):
    interactions=[f for f in feature_list if '×' in f]
    operand=[]
    for f in interactions:
        l,r=[s.strip() for s in f.split('×')]
        for op in (l,r):
            if op not in feature_list and op not in operand:
                operand.append(op)
    feats=list(feature_list)+operand
    parts=[]
    # calendar & holiday
    cal_keys=['month','quarter','month_sin','month_cos','holiday_ratio','holiday_flag']
    if any(f in cal_keys for f in feats):
        cal=generate_calendar_features(df,feature_list=feats)
        keep=[f for f in cal_keys if f in feats]
        parts.append(cal[keep])
    # group
    if 'group_id' in feats:
        parts.append(generate_group_id(df,grouping))
    # lag
    lag_map={}
    for f in feats:
        m=re.fullmatch(r'(?P<c>\w+)_lag_(?P<l>\d+)',f)
        if m: lag_map.setdefault(m.group('c'),[]).append(int(m.group('l')))
    if lag_map: parts.append(generate_lag_features(df,lag_map))
    # roll mean/std
    roll_map={}
    for f in feats:
        m=re.fullmatch(r'(?P<c>\w+)_rolling_(?P<s>mean|std)_(?P<w>\d+)',f)
        if m: roll_map.setdefault(m.group('c'),{}).setdefault(m.group('s'),[]).append(int(m.group('w')))
    if roll_map: parts.append(generate_rolling_features(df,roll_map))
    # expanding
    exp_map={}
    for f in feats:
        m=re.fullmatch(r'(?P<c>\w+)_expanding_(?P<s>mean|std)',f)
        if m: exp_map.setdefault(m.group('c'),[]).append(m.group('s'))
    if exp_map: parts.append(generate_expanding_features(df,exp_map))
    # intervals
    if any(f in ['periods_since_last_demand','zero_run_length','avg_interarrival'] for f in feats):
        parts.append(generate_interval_features(df))
    # rolling counts
    if any('_rolling_count_' in f for f in feats):
        cnt_map={}
        for f in feats:
            m=re.fullmatch(r'(?P<c>\w+)_rolling_count_(?P<w>\d+)',f)
            if m: cnt_map.setdefault(m.group('c'),[]).append(int(m.group('w')))
        parts.append(generate_rolling_count_features(df,cnt_map))
    #pmi
    if 'pmi_index' in feats:
        parts.append(generate_pmi_feature(df))
    # last positive
    if any(f.startswith('last_') for f in feats):
        n_list=[]
        for f in feats:
            m=re.fullmatch(r'last_(?P<n>\d+)_positive_(?:mean|median)',f)
            if m: n_list.append(int(m.group('n')))
        if 'last_positive_demand' in feats: n_list.append(1)
        parts.append(generate_last_positive_features(df,sorted(set(n_list))))
    # raw
    existing=pd.concat(parts,axis=1).columns if parts else []
    raw=[f for f in feats if f not in existing and '×' not in f]
    if raw: parts.append(df[raw])
    # build X
    X=pd.concat(parts,axis=1) if parts else pd.DataFrame(index=df.index)
    # interactions
    if interactions:
        X=pd.concat([X,generate_interaction_features(X,interactions)],axis=1)
    return X
