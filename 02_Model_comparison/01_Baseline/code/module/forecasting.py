import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV



# ---------------------------
# BASE FORECASTING METHODS
# ---------------------------

def superpessimist_forecast(series, forecast_horizon):
    # Als "Baseline", gibt einfach immer Null zurück
    # returned immer den forecast, und ob es ein Fallback war (also Fallback heisst, dass es nur den forecast gab, weil z.B. die daten nicht gereicht haben
    return np.zeros(forecast_horizon), True

def croston_forecast(series, forecast_horizon, alpha=0.1):
    values = series.values
    index_first_nonzero = np.argmax(values > 0)
    # argmax gibt 0 zurück, wenn alle Werte 0 oder kleiner sind, daher check ob der wert bei dem index tatsächlich 0 ist
    if values[index_first_nonzero] == 0:
        return np.zeros(forecast_horizon), True

    indizes_nonzero = np.nonzero(values > 0)[0]
    # berechne initiales Intervall aus den ersten beiden Nicht‑Null‑Terminen
    if len(indizes_nonzero) > 1:
        initial_interval = indizes_nonzero[1] - indizes_nonzero[0]
    else:
        # nur ein einziger Nicht‑Null‑Termin im Sample → Intervall = 1
        initial_interval = 1

    # Croston werte initalisieren
    smoothed_demandsize, smoothed_interval, nullperioden_zähler = values[index_first_nonzero], initial_interval, 1
    for x in values[index_first_nonzero+1:]:
        if x > 0:
            smoothed_demandsize += alpha * (x - smoothed_demandsize)
            smoothed_interval += alpha * (nullperioden_zähler - smoothed_interval)
            nullperioden_zähler = 1
        else:
            nullperioden_zähler += 1
    return np.full(forecast_horizon, smoothed_demandsize / smoothed_interval), False


def tsb_forecast(series, forecast_horizon, alpha=0.1):
    # "naive" implementierung mit nur einem smoothing param alpha, ggf. noch alpha und beta nehmen falls tsb vielversprechend ist
    values = series.values
    index_first_nonzero = np.argmax(values > 0)
    if values[index_first_nonzero] == 0:
        return np.zeros(forecast_horizon), True

    # werte initialisieren
    smoothed_demand_probability, smoothed_demand_size = 1.0, values[index_first_nonzero]
    for x in values[index_first_nonzero+1:]:
        nachfrage_binary = int(x > 0) # also 1 oder 0, je nachdem obs in der periode nachfrage gibt
        smoothed_demand_probability += alpha * (nachfrage_binary - smoothed_demand_probability)
        if nachfrage_binary:
            smoothed_demand_size += alpha * (x - smoothed_demand_size)
    return np.full(forecast_horizon, smoothed_demand_probability * smoothed_demand_size), False


def sba_forecast(series, forecast_horizon, alpha=0.1):
    values = series.values
    index_first_nonzero = np.argmax(values > 0)
    if values[index_first_nonzero] == 0:
        return np.zeros(forecast_horizon), True

    indizes_nonzero = np.nonzero(values > 0)[0]
    # berechne initiales Intervall aus den ersten beiden Nicht‑Null‑Terminen
    if len(indizes_nonzero) > 1:
        initial_interval = indizes_nonzero[1] - indizes_nonzero[0]
    else:
        # nur ein einziger Nicht‑Null‑Termin im Sample → Intervall = 1
        initial_interval = 1

    # werte quasi gleich wie bei croston initialisieren
    smoothed_demand, smoothed_interval, nullperioden_zähler = values[index_first_nonzero], initial_interval, 1
    for x in values[index_first_nonzero+1:]:
        if x > 0:
            smoothed_demand += alpha * (x - smoothed_demand)
            smoothed_interval += alpha * (nullperioden_zähler - smoothed_interval)
            nullperioden_zähler = 1
        else:
            nullperioden_zähler += 1
    # also eig genau gleich wie croston, nur hier kommt jetzt noch der korrekturfaktor davor
    return np.full(forecast_horizon, (1 - alpha/2) * (smoothed_demand / smoothed_interval)), False

def sbj_forecast(series, forecast_horizon, alpha=0.1):
    values = series.values
    index_first_nonzero = np.argmax(values > 0)
    if values[index_first_nonzero] == 0:
        return np.zeros(forecast_horizon), True

    indizes_nonzero = np.nonzero(values > 0)[0]
    # berechne initiales Intervall aus den ersten beiden Nicht‑Null‑Terminen
    if len(indizes_nonzero) > 1:
        initial_interval = indizes_nonzero[1] - indizes_nonzero[0]
    else:
        # nur ein einziger Nicht‑Null‑Termin im Sample → Intervall = 1
        initial_interval = 1

    # werte quasi gleich wie bei croston initialisieren
    smoothed_demand, smoothed_interval, nullperioden_zähler = values[index_first_nonzero], initial_interval, 1
    for x in values[index_first_nonzero + 1:]:
        if x > 0:
            smoothed_demand += alpha * (x - smoothed_demand)
            smoothed_interval += alpha * (nullperioden_zähler - smoothed_interval)
            nullperioden_zähler = 1
        else:
            nullperioden_zähler += 1
    # also eig genau gleich wie croston, nur hier kommt jetzt noch der korrekturfaktor für sbj davor
    corr = 1 - alpha / (2 - alpha)
    return np.full(forecast_horizon, corr * (smoothed_demand / smoothed_interval)), False # dieses false am ende hier immer noch, weil es ja NICHT fallback is

def ewma_forecast(series, forecast_horizon, alpha=0.1):
    # also single exponential smoothing (also nur schätzung von level)
    # das hier ist wieder das mit dem "True", wenn die Daten nicht gereicht haben und es einen baseline forecast (also einfach nur 0) zurückgibt
    if series.sum() == 0:
        return np.zeros(forecast_horizon), True
    # gibt einfach ne funktion die das macht

    #wieso hier mean und iloc(-1): mean muss man (wieso auch immer) callen, damit man überhaupt den forecast bekommt, ohne mean bekommt man so ein ewma object
    # und dann iloc(-1), um eben den letzten wert zu haben, den wir dann entsprechend als aktuellen forecast nutzen
    last = series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    return np.full(forecast_horizon, last), False

def holt_forecast(series, forecast_horizon, alpha=0.1, beta=0.1):
    # also double exponential smoothing (es wird also sowohl level, als auch trend geschätzt)
    values = series.values
    if len(values) == 0 or (values == 0).all():
        return np.zeros(forecast_horizon), True

    # ersten wert als base-level nehmen
    level_estimate = values[0]

    # initial trend estimate
    trend_estimate = values[1] - values[0] if len(values) > 1 else 0.0

    for x in values[1:]:
        previous_level_estimate = level_estimate
        #level estimate updaten
        level_estimate = alpha * x + (1 - alpha) * (level_estimate + trend_estimate)
        # trend estimate updaten
        trend_estimate = beta * (level_estimate - previous_level_estimate) + (1 - beta) * trend_estimate

    
    period = np.arange(1, forecast_horizon + 1)
    # hier wird der forecast zurückgegeben, also level + trend * period, nach dem trend über den schätzzeitraum erhöht, deshalb die period variable
    return level_estimate + trend_estimate * period, False


def holt_winters_forecast(series, forecast_horizon,
                          alpha=0.1, beta=0.1, gamma=0.1,
                          seasonal_periods=12, seasonal='add'):
    # also triple exponential smoothig (level, trend und saison)
    values = series.values
    number_of_observations, saisonzyklus_länge = len(values), seasonal_periods

    # wir setzen mal mindestens zwei volle saisonzyklen (also jahre) voraus, ansonsten ist die schätzung nicht sinnvoll
    # (vllt das hier auch rausnehmen, damit es nicht zu viele fallbacks gibt)
    if number_of_observations < 2 * saisonzyklus_länge:
        return np.zeros(forecast_horizon), True

    #initialisierung der estimates
    # level estimate (durchschnitt des ersten jahres)
    level_estimate = values[:saisonzyklus_länge].mean()

    # trend estimate (differenz aus punkt zu dem punkt ein jahr zuvor, geteilt durch 12 (für die monate) und dann den schnitt daraus
    trend_estimate = ((values[saisonzyklus_länge:] - values[:-saisonzyklus_länge]) / saisonzyklus_länge).mean()


    # saisonalität estimate (abweichung vom level estimate)
    if seasonal == 'add':
        seasonal_deviations = values[:saisonzyklus_länge] - level_estimate
    else:
        seasonal_deviations = values[:saisonzyklus_länge] / level_estimate

    #result array initialisieren mit nullen erstmal
    result = np.zeros(number_of_observations + forecast_horizon)

    # rekursives smoothing über die historischen daten
    for t in range(number_of_observations):
        # aktuellen saisonalen faktor holen
        season_factor = seasonal_deviations[t % saisonzyklus_länge]

        if seasonal == 'add':
            result[t] = level_estimate + trend_estimate + season_factor
        else:
            result[t] = (level_estimate + trend_estimate) * season_factor

        current_observation = values[t]

        # level estimate updaten
        if seasonal == 'add':
            updated_level = alpha * (current_observation - season_factor) + (1 - alpha) * (level_estimate + trend_estimate)
        else:
            updated_level = alpha * (current_observation / season_factor) + (1 - alpha) * (level_estimate + trend_estimate)

        # trend estimate updaten
        updated_trend = beta * (updated_level - level_estimate) + (1 - beta) * trend_estimate

        # saisonfaktor updaten
        if seasonal == 'add':
            seasonal_deviations[t % saisonzyklus_länge] = gamma * (current_observation - updated_level) + (1 - gamma) * season_factor
        else:
            seasonal_deviations[t % saisonzyklus_länge] = gamma * (current_observation / updated_level) + (1 - gamma) * season_factor

        # neue levels und trends setzen
        level_estimate, trend_estimate = updated_level, updated_trend

    # vorhersage für den forecast horizon
    for h in range(1, forecast_horizon + 1):
        season_factor = seasonal_deviations[(number_of_observations + h - 1) % saisonzyklus_länge]

        # ähnlich wie vorher mit dem trend aufrechnen, nur zusätlich auch der seasonal factor
        if seasonal == 'add':
            result[number_of_observations + h - 1] = level_estimate + h * trend_estimate + season_factor
        else:
            result[number_of_observations + h - 1] = (level_estimate + h * trend_estimate) * season_factor

    # nur die forcast horizon werte zurückgeben ( in results sind ja noch die historischen geglätteten werte drin)
    forecasts = result[number_of_observations:]
    return forecasts, False


def poisson_forecast(series, forecast_horizon, quantile=None):
    values = np.asarray(series)
    if values.size == 0 or (values == 0).all():
        return np.zeros(forecast_horizon), True

    lambbda = values.mean() # bennenung : lambbda weil man nicht lambda nennen darf, ist der parameter für die poissonverteilung
    if quantile is None:
        # wenn kein quantil angegeben ist, dann einfach nur den schnitt
        return np.full(forecast_horizon, lambbda), False
    # wenn quantil angegeben ist, dann den quantilwert zurückgeben (basierend auf der normalverteilung) -> hier dann noch für verschiedene quantile testen!
    return np.full(forecast_horizon, poisson(mu=lambbda).ppf(quantile)), False


def nbd_forecast(series, forecast_horizon, quantile=None):
    values = np.asarray(series)
    if values.size == 0 or (values == 0).all():
        return np.zeros(forecast_horizon), True

    lambbda, varianz = values.mean(), values.var(ddof=0) # also lambbda hier wieder der erwartungswert
    # NBD is ja insbesondere gut, wenn man überdispersion hat, wenn das also garnicht der fall ist, nehmen wir normal poisson
    if varianz <= lambbda:
        if quantile is None:
            return np.full(forecast_horizon, lambbda), False
        return np.full(forecast_horizon, poisson(mu=lambbda).ppf(quantile)), False

    # die parameter für die negative binomial distribution aus den historischen daten schätzen
    r = lambbda**2 / (varianz - lambbda)
    p = r / (r + lambbda)
    if quantile is None:
        # wenn kein quantil angegeben ist, dann geben wir auch einfach den erwartungswert zurück (also dann verhalten sich nbd und poisson exakt gleich,
        # weil sich ja nur die verteilung unterscheidet)
        return np.full(forecast_horizon, lambbda), False
    return np.full(forecast_horizon, nbinom(n=r, p=p).ppf(quantile)), False



def zip_forecast(series, forecast_horizon, quantile=None):
    #zero inflated poisson!
    values = np.asarray(series)
    if values.size == 0 or (values == 0).all():
        return np.zeros(forecast_horizon), True


    lambbda = values.mean()

    #values==0 gibt uns ein array zurück, wo die werte 1 sind, wenn der wert 0 ist und 0 wenn der wert > 0 ist
    #np.mean darüber gibt uns dann den anteil der nullen zurück
    ratio_of_zeros = np.mean(values == 0)

    expn = np.exp(-lambbda)
    #pi: mit wahrschienlichkeit pi erzeugt die beobachtung wert 0 und mit wahrscheinlichkeit 1-pi erzeugt sie eine poissonverteilung mit parameter lambbda
    pi = max(0.0, (ratio_of_zeros - expn) / (1 - expn)) if expn < ratio_of_zeros else 0.0
    if quantile is None:
        return np.full(forecast_horizon, (1 - pi) * lambbda), False
    poisson_dist = poisson(mu=lambbda)
    #quantil anpassen wegen der pi wahrscheinlichkeit (also der zero inflation)
    adjusted_poisson_quantile = np.clip((quantile - pi) / (1 - pi), 0, 1)
    return np.full(forecast_horizon, poisson_dist.ppf(adjusted_poisson_quantile)), False


def nhpp_poisson_forecast(series, forecast_horizon,
                          method='ewma', alpha=0.1, window=None, quantile=None):
    # nicht homogener poisson prozess (also das aktuelle lambda wird adaptiv geschätzt)
    series = pd.Series(series).astype(float) # wichtig, dass es float ist
    if series.empty or series.sum() == 0:
        return np.zeros(forecast_horizon), True

    if method == 'ewma':
        # ganz normal exponentiell geglättet
        lambda_estimate = series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        #mit rolling window (window muss man angeben)
    elif method == 'rolling' and window:
        # mit rolling window (window muss man angeben)
        lambda_estimate = series.rolling(window, min_periods=1).mean().iloc[-1]
    else:
        # einfach durchscnitt, also gleich wie normaler poisson
        lambda_estimate = series.mean()
    if quantile is None:
        return np.full(forecast_horizon, lambda_estimate), False
    return np.full(forecast_horizon, poisson(mu=lambda_estimate).ppf(quantile)), False

# ---------------------------
# WILLEMAIN BOOTSTRAP
# ---------------------------

def compute_markov_probs(series):
    binary_series = np.where(series > 0, 1, 0)
    n0 = n1 = to0 = to1 = 0
    for i in range(len(binary_series) - 1):
        if binary_series[i] == 0:
            n0 += 1
            if binary_series[i+1] == 0: to0 += 1
        else:
            n1 += 1
            if binary_series[i+1] == 1: to1 += 1
    p01 = 1 - to0/n0 if n0 > 0 else 0.1 # mit default werten, falls es sich aus irgendeinem grund nicht berechnen lässt
    p11 = to1/n1 if n1 > 0 else 0.9
    return p01, p11

def jitter_value(x, jitter_factor=0.1):
    noise = np.random.normal(0, jitter_factor * x)
    j = int(round(x + noise))
    return j if j >= 0 else x

def willemain_boostrap_pointforecast_multistep_mean(series, forecast_horizon,
                                                   n_bootstrap=1000, central_tendency='mean'):
    values = np.array(series)
    if (values == 0).all():
        return np.zeros(forecast_horizon), True


    nonzero_values = values[values > 0]
    p01, p11 = compute_markov_probs(values)
    # letzte Periode als Startwert
    state = 1 if values[-1] > 0 else 0
    paths = np.zeros((n_bootstrap, forecast_horizon)) # n_bootstrap x forecast_horizon matrix mit nullen gefüllt
    # iteriere über die zeilen (also bootstrap samples)
    for i in range(n_bootstrap):
        current_state = state
        # pfad den der markov prozess für dieses boostrap sample "geht"
        path = np.zeros(forecast_horizon)
        # dann "über den pfad" gehen
        for t in range(forecast_horizon):
            # random state transition
            next_state = 1 if np.random.rand() < (p01 if current_state == 0 else p11) else 0
            val = jitter_value(np.random.choice(nonzero_values)) if next_state == 1 else 0
            path[t] = val
            current_state = next_state
        # das hier heisst, wir haben dann für jede periode den in DIESER PERIODE erwarteten wert, nicht kumuliert oderso
        paths[i] = path
    if central_tendency == 'median':
        prognose = np.median(paths, axis=0)
    elif central_tendency == 'mean':
        prognose = np.mean(paths, axis=0)
    else:
        prognose = np.quantile(paths, central_tendency, axis=0)
    return prognose, False

# ---------------------------
# RECURSIVE FORECAST HELPER
# ---------------------------

def recursive_forecast_base(model_func, history, forecast_horizon, **kwargs):
    # das hier ist ein wrapper, mit dem wir für alle modelle den forecast rekursiv machen können (also immer nur eine periode voraus,
    # und dann darauf wieder die nächste periode voraus)
    # wir tracken die forecasts in einer liste und wichtig: die fallbacks auch! (damit wir dann später wissen, ob die methode nur gut war weil sie
    # immer 0 zurückgegeben hat)
    forecast_list, fallback_list = [], []
    current_history = history.copy()
    for _ in range(forecast_horizon):
        fc_step, is_fb = model_func(current_history, forecast_horizon=1, **kwargs)
        fc = fc_step[0]
        forecast_list.append(fc)
        fallback_list.append(is_fb)
        # the current history updates, so now the latest forecast is part of the history
        current_history = pd.concat([current_history, pd.Series([fc])], ignore_index=True)

    # returnen der forecast und der fallback liste (die fallback liste enthält also nur booleans, später am besten dann alles als fallback sehen,
    # das mehr als 0 fallbacks hatte, wir haben hier ja pro forecast horizon step einen fallback wert)
    return np.array(forecast_list), np.array(fallback_list, dtype=bool)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

def fill_missing_feature(series):
    # f = series.ffill().bfill().fillna(0) # das hatte ich davor, noch überlegen, ob ich das vllt stattdessen drin lass (je nachdem, oder
    # ne intelligentere fill methode bauen wenn Zeit ist)
    f = series.fillna("MISSING_VALUE")
    return f.infer_objects(copy=False)

def create_features(df, lag, extra_features=[]):
    #basierend auf dem lag und den extra features, die wir haben, erstellen wir die features für einen dataframe
    df_feat = df.copy()
    for l in range(1, lag+1):
        df_feat[f"lag_{l}"] = df_feat["demand"].shift(l)
    for feat in extra_features:
        if feat in df_feat.columns:
            df_feat[feat] = fill_missing_feature(df_feat[feat])
            if df_feat[feat].dtype == object:
                # factorize heisst, das feature in eine integer repräsentation umwandeln, damit wir es auf jeden fall als feature nutzen können
                df_feat[feat] = pd.factorize(df_feat[feat])[0]
    return df_feat.dropna()


# ---------------------------
# FEATURE VARIANTS
# ---------------------------


def create_features_variant_base(df, extra_features=[]):
    # einfach nur lags bis lag 6 und passed extra features
    return create_features(df, lag=6, extra_features=extra_features)

def create_features_variant_a(df, extra_features=[]):
    # dasselbe plus lag_12
    df2 = create_features(df, lag=6, extra_features=extra_features)
    df2["lag_12"] = df["demand"].shift(12)
    return df2.dropna()

def create_features_variant_b(df, extra_features=[], period=12, K=2):
    # mit fourier features für seasonality, aber sinnhaftigkeit etwas fragwürdig weil wir nicht so richtig starke seasonality haben,
    # trotzdem zum Testen
    df2 = create_features(df, lag=6, extra_features=extra_features)
    t = np.arange(len(df2))
    for k in range(1, K+1):
        df2[f"fourier_sin_{k}"] = np.sin(2*np.pi*k*t/period)
        df2[f"fourier_cos_{k}"] = np.cos(2*np.pi*k*t/period)
    return df2

def create_features_variant_c(df, extra_features=[], period=12, K=2, rolling_window=3):
    df2 = create_features_variant_a(df, extra_features)
    t = np.arange(len(df2))
    for k in range(1, K+1):
        df2[f"fourier_sin_{k}"] = np.sin(2*np.pi*k*t/period)
        df2[f"fourier_cos_{k}"] = np.cos(2*np.pi*k*t/period)
    # bis hier gleich wie b

    interval = []
    cnt = 0
    # zähler perioden seit letzter nachfrage
    for x in df2["demand"]:
        cnt = 0 if x > 0 else cnt+1
        interval.append(cnt)
    df2["inter_demand_interval"] = interval
    df2["rolling_mean"] = df2["demand"].rolling(window=rolling_window, min_periods=1).mean()
    return df2.dropna()

def create_features_variant_d(df, extra_features=[]):
    df2 = create_features_variant_base(df, extra_features)
    df2.index = pd.to_datetime(df2.index)
    m = df2.index.month
    #einfach nur base plus one-hot-encoding der monate
    return df2.join(pd.get_dummies(m, prefix="month", drop_first=False))

def create_features_variant_e(df, extra_features=[]):
    df2 = create_features_variant_base(df, extra_features)
    # anteil der zero-werte in den letzten 3 perioden
    df2["rolling_zero_prop"] = df2["demand"].rolling(window=3, min_periods=1).apply(lambda x: np.mean(x==0), raw=True)
    # standardabweichung der letzten drei werte
    df2["rolling_std"]      = df2["demand"].rolling(window=3, min_periods=1).std(ddof=0)
    return df2.dropna()


def create_features_variant_f(df, extra_features=[]):
    #print(f"[DEBUG f] eingehende Serie hat len={len(df)}, columns={list(df.columns)}")
    # very rich feature set for GBM
    # wenn history insgesamt weniger als 6 monate hat, einfach auf base-features zurückfallen
    lookback = 6
    # seasonaler zuschlag erst ab 12 perioden
    seasonal = 12
    if len(df) < lookback:
        return create_features_variant_base(df, extra_features)

    # immer erstmal die 1–6 lags und extra_features holen
    df2 = create_features(df, lag=lookback, extra_features=extra_features)

    # wenn genug daten für den 12-monat-lag da sind, add lag_12
    if len(df) >= seasonal + 1:
        df2["lag_12"] = df["demand"].shift(seasonal)

    # jetzt fourier-features nur wenn >= seasonal
    if len(df) >= seasonal:
        t = np.arange(len(df2))
        for k in range(1, 4):  # K=3
            df2[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / seasonal)
            df2[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / seasonal)

    # perioden seit letzter nachfrage – geht immer
    interval = []
    cnt = 0
    for x in df2["demand"]:
        cnt = 0 if x > 0 else cnt + 1
        interval.append(cnt)
    df2["inter_demand_interval"] = interval

    # rolling window stats – hier min_periods=1, also klappt schon ab einer perode
    rolling_window = 6
    df2["rolling_mean"] = df2["demand"].rolling(window=rolling_window, min_periods=1).mean()
    df2 = df2.dropna()  # wirft nur die ganz ersten lookback nans weg


    # monat dummies
    monatsserie = pd.Series(df2.index.month, index=df2.index)
    df2 = df2.join(pd.get_dummies(monatsserie, prefix="month", drop_first=False))
    # quartal dummies
    quartalsserie = pd.Series(df2.index.quarter, index=df2.index)
    df2 = df2.join(pd.get_dummies(quartalsserie, prefix="quarter", drop_first=False))
    # ungefährer feiertagsanteil in deutschland (weltweit auch überlegt, aber da ist dann einfach fast jeder monat 3%, und das bringt dann ja nix
    holiday_ratio = {
        1: 3.2, 2: 0.0, 3: 1.6, 4: 5.0,
        5: 6.5, 6: 3.3, 7: 0.0, 8: 0.6,
        9: 0.0, 10: 3.2, 11: 1.7, 12: 6.5
    }
    # die holiday ratio für den monat
    df2.index = pd.to_datetime(df2.index)
    df2["holiday_ratio"] = df2.index.month.map(lambda m: holiday_ratio.get(m, 0.0) / 100.0)

    # adaptive fenstergrösse: rolling mit min_periods=1 damit wir nicht zu viele naNs kriegen
    # zusätzliche rolling stats
    df2["rolling_mean_6"]  = df2["demand"].rolling(6,  min_periods=1).mean().fillna(0)
    df2["rolling_mean_12"] = df2["demand"].rolling(12, min_periods=1).mean().fillna(0)

    df2["rolling_std_3"]   = df2["demand"].rolling(3,  min_periods=1).std().fillna(0)
    df2["rolling_std_6"]   = df2["demand"].rolling(6,  min_periods=1).std().fillna(0)
    df2["rolling_std_12"]  = df2["demand"].rolling(12, min_periods=1).std().fillna(0)

    df2["rolling_skew_3"]  = df2["demand"].rolling(3,  min_periods=1).skew().fillna(0)
    df2["rolling_skew_6"]  = df2["demand"].rolling(6,  min_periods=1).skew().fillna(0)
    df2["rolling_skew_12"] = df2["demand"].rolling(12, min_periods=1).skew().fillna(0)

    df2["rolling_kurtosis_3"]  = df2["demand"].rolling(3,  min_periods=1).kurt().fillna(0)
    df2["rolling_kurtosis_6"]  = df2["demand"].rolling(6,  min_periods=1).kurt().fillna(0)
    df2["rolling_kurtosis_12"] = df2["demand"].rolling(12, min_periods=1).kurt().fillna(0)

    df2["rolling_zero_prop_3"]  = df2["demand"].rolling(3,  min_periods=1).apply(lambda x: np.mean(x == 0), raw=True)
    df2["rolling_zero_prop_6"]  = df2["demand"].rolling(6,  min_periods=1).apply(lambda x: np.mean(x == 0), raw=True)
    df2["rolling_zero_prop_12"] = df2["demand"].rolling(12, min_periods=1).apply(lambda x: np.mean(x == 0), raw=True)
    # markov probs
    p01, p11 = compute_markov_probs(df["demand"])
    df2["p01"], df2["p11"] = p01, p11
    # txn_count (also anzahl der transaktionen für den monat) und unique_recipients direkt mit reinnehmen
    df2["txn_count"] = df["txn_count"]
    df2["unique_recipients"] = df["unique_recipients"]

    # werk und bewegungsart als kategorische features (fill missing + factorize)
    df2["Werk"] = df["Werk"].fillna("MISSING")
    df2["Werk"] = pd.factorize(df2["Werk"])[0]
    df2["Bewegungsart"] = df["Bewegungsart"].fillna("MISSING")
    df2["Bewegungsart"] = pd.factorize(df2["Bewegungsart"])[0]

    df2_clean = df2.dropna()
    #print(f"[DEBUG f] raus aus f: df2_clean.shape = {df2_clean.shape}")
    return df2_clean

# ---------------------------
# GROUP‑WIDE RF‑HYBRID
# ---------------------------

def _get_variant_config(variant):
    # helferfunktion um das setup der variante zu bekommen
    if variant == "base":
        return create_features_variant_base, 6, croston_forecast, "rf"
    if variant == "a":
        return create_features_variant_a, 12, croston_forecast, "rf"
    if variant == "b":
        return create_features_variant_b, 6, croston_forecast, "rf"
    if variant == "c":
        return create_features_variant_c, 12, croston_forecast, "rf"
    if variant == "d":
        return create_features_variant_d, 6, croston_forecast, "rf"
    if variant == "e":
        return create_features_variant_e, 6, croston_forecast, "rf"
    if variant == "c_tuned":
        return create_features_variant_c, 12, croston_forecast, "rf_tuned"
    if variant == "c_ewma":
        return create_features_variant_c, 12, ewma_forecast, "rf"
    if variant == "c_svr_ewma":
        return create_features_variant_c, 12, ewma_forecast, "svr"
    raise ValueError(f"Unknown variant '{variant}'")

def train_hybrid_variant_group(train_dfs, variant, extra_features=[], random_state=42):
    # trainiert hybrid modell über mehrere zeitreihen (also die materialien innerhalb einer gruppe dann z.B.)
    # mit Random forest, tuned random forest oder SVR, je nach reg_type in der config
    feat_fn, lag, baseline_fn, reg_type = _get_variant_config(variant)
    X_parts, y_parts = [], []
    for df in train_dfs:


        feats   = feat_fn(df, extra_features)
        roll_fc, _ = recursive_forecast_base(baseline_fn, df["demand"], lag, alpha=0.1)

        n_feat = len(feats)
        if n_feat < len(roll_fc):
            aligned = roll_fc[-n_feat:]
        else:
            aligned = roll_fc
        feats = feats.iloc[-len(aligned):].copy()
        feats["baseline_fc"] = aligned


        feats["residual"]    = feats["demand"] - feats["baseline_fc"]
        X_parts.append(feats.drop(columns=["demand", "residual"]))
        y_parts.append(feats["residual"])
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)

    if reg_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X, y)
    elif reg_type == "rf_tuned":
        grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        gs = GridSearchCV(RandomForestRegressor(random_state=random_state), grid,
                          cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(X, y)
        model = gs.best_estimator_
    else:
        model = SVR(kernel="rbf")
        model.fit(X, y)

    return {
        "feat_fn": feat_fn,
        "effective_lag": lag,
        "baseline_func": baseline_fn,
        "regressor": model,
        "extra_features": extra_features,
        "feature_names": X.columns.tolist()
    }

def recursive_rf_hybrid_forecast(pretrained_model, training_data, forecast_horizon):
    # zum durchführen eines rekursiven Mehrschritt forecasts
    # hab die hier umbenannt, das man es besser begreift, was jetzt was ist-> TODO: auch bei den anderen funktionen so machen
    feature_extractor      = pretrained_model["feat_fn"]
    lookback               = pretrained_model["effective_lag"]
    base_forecaster        = pretrained_model["baseline_func"]
    residual_regressor     = pretrained_model["regressor"]
    extra_feature_names    = pretrained_model["extra_features"]
    feature_columns        = pretrained_model["feature_names"]

    demand_history = training_data["demand"].copy()
    # speichere die letzten extra features ab, weil wir ja für unsere predicted periods keine features haben
    last_extra_values = {
        name: training_data.iloc[-1].get(name, 0)
        for name in extra_feature_names
    }

    predictions   = []
    fallback_flags = []
    for _ in range(forecast_horizon):
        base_fc_array, base_was_fallback_arr = base_forecaster(
            demand_history, forecast_horizon=1, alpha=0.1
        )
        # base forecast für die nächste periode holen
        base_value          = base_fc_array[0]
        # merken, ob der base forecast einen fallback hatte
        base_was_fallback   = bool(base_was_fallback_arr)

        # kontext der letzten lookback perioden plus base forecast
        context_df = demand_history.iloc[-lookback:].to_frame("demand")
        last_date  = demand_history.index[-1]
        next_date  = last_date + pd.DateOffset(months=1)
        augmented_df = pd.concat(
            [context_df, pd.DataFrame({"demand": [base_value]}, index=[next_date])],
            axis=0
        )
        # das hier ist naiv/einfach gehalten: einfach die extra features fortführen
        for name in extra_feature_names:
            augmented_df[name] = last_extra_values[name]

        # der forecast extractor gibt uns einen features df
        feature_matrix = feature_extractor(augmented_df, extra_feature_names)
        if feature_matrix.empty:
            # wenn wir keine features haben, ist das auch ein fallback, dann nehmen wir einfach den base forecast
            # Frage: sollten wir hier nicht vllt auch einfach 0 nehmen?
            prediction    = base_value
            used_fallback = True
        else:
            # den base forecast der feature matrix anhängen
            feature_matrix.loc[feature_matrix.index[-1], "baseline_fc"] = base_value
            X_new = (
                feature_matrix
                .iloc[[-1]]
                .drop(columns=["demand"])
                .reindex(columns=feature_columns, fill_value=0)
            )
            # dann nutzen wir den vortrainierten residual regressor um das residual zu predicten
            residual    = residual_regressor.predict(X_new)[0]
            # keine null prediction
            prediction  = max(base_value + residual, 0)
            used_fallback = False

        # die prediction für diese forecast horizon periode den predictions anhängen
        predictions.append(prediction)
        # merken, ob wir einen fallback hatten (entweder in base oder wegen feature creation)
        fallback_flags.append(base_was_fallback or used_fallback)
        # der history die aktuelle prediction anhängen
        last_date = demand_history.index[-1]
        next_date = last_date + pd.DateOffset(months=1)
        demand_history = pd.concat(
            [demand_history, pd.Series([prediction], index=[next_date])],
            axis=0
        )
        # und dann weiter in der schleife zur nächsten forecast horizon periode
    return np.array(predictions), np.array(fallback_flags, dtype=bool)




# ---------------------------
# IM FOLGENDEN SPEZIALISIERT AUF GBM, EINMAL "DIRECT" UND EINMAL "HYBRID"
# ---------------------------

# ---------------------------
# DIRECT GBM WITH BASE FEATURES
# ---------------------------

def train_direct_gbm_base_group(group_train_dfs, extra_features=None, lookback=6, random_state=42):
    # debug: Länge aller übergebenen train_dfs
    lengths = [len(df) for df in group_train_dfs]
    print(f"→ alle train_dfs haben Längen min={min(lengths)} max={max(lengths)}")
    if extra_features is None:
        extra_features = []
    feature_parts = []
    target_parts  = []
    for series_df in group_train_dfs:
        feature_matrix = create_features_variant_f(series_df, extra_features)
        # demand column natürlich droppen, darauf wollen wir ja trainieren
        feature_parts.append(feature_matrix.drop(columns=["demand"]))
        # daher ist das dann im targets_part
        target_parts.append(feature_matrix["demand"])
    # über alle serien zu einem trainingsset zusammenführen
    X_train = pd.concat(feature_parts, ignore_index=True)
    y_train = pd.concat(target_parts,  ignore_index=True)
    # gbm auf den reinen demand trainen
    gbm_model = GradientBoostingRegressor(random_state=random_state)
    gbm_model.fit(X_train, y_train)
    return {
        "model":             gbm_model,
        "lag":               lookback,
        "extra_features":    extra_features,
        "feature_extractor": create_features_variant_f,
        "feature_names":     X_train.columns.tolist()
    }


def recursive_direct_gbm_forecast_pretrained_base(pretrained_model, training_df, forecast_horizon):
    # wie gewohnt rekursiv trainieren mit dem pre-trained model auf der group
    gbm_model           = pretrained_model["model"]
    lookback            = pretrained_model["lag"]
    extra_feature_names = pretrained_model["extra_features"]
    feat_fn             = pretrained_model["feature_extractor"]
    feature_names       = pretrained_model["feature_names"]

    demand_history = training_df["demand"].copy()
    """
    # Lassen wir mal raus, schränkt das ganze unnötig ein eigentlich
    if len(demand_history) < lookback + 1:
        croston_fc, _ = croston_forecast(demand_history, forecast_horizon)
        # das stellt ja einen Fallback auf Croston dar, TODO: bin mir hier auch unsicher, ob wir einen fallback auf croston oder auf superpessimist
        # machen sollten
        fallback_flags = np.ones(forecast_horizon, dtype=bool)
        return croston_fc, fallback_flags
    """

    # wieder die letzten werte der features sichern
    last_extra_values = {
        name: training_df.iloc[-1].get(name, 0)
        for name in extra_feature_names
    }

    predictions    = []
    fallback_flags = []
    # das hier jetzt sehr analog zu dem oben
    for _ in range(forecast_horizon):
        context_df = demand_history.iloc[-lookback:].to_frame("demand")
        last_date  = demand_history.index[-1]
        next_date  = last_date + pd.DateOffset(months=1)
        dummy_val  = predictions[-1] if predictions else demand_history.iloc[-1]
        context_df = pd.concat(
            [context_df, pd.DataFrame({"demand": [dummy_val]}, index=[next_date])],
            axis=0
        )
        for name in extra_feature_names:
            context_df[name] = last_extra_values[name]

        # features bauen
        feature_matrix = feat_fn(context_df, extra_feature_names)
        if feature_matrix.empty:
            prediction    = 0.0
            used_fallback = True
        else:
            # bereits existing model nutzen zum forecasten
            X_new         = feature_matrix.iloc[[-1]].drop(columns=["demand"])
            X_new         = X_new.reindex(columns=feature_names, fill_value=0)
            prediction    = max(gbm_model.predict(X_new)[0], 0)
            used_fallback = False

        predictions.append(prediction)
        fallback_flags.append(used_fallback)

        # und wie gewohnt eben einmal über den forecast horizon
        # und vorhersage als neuer demand history eintrag
        last_date = demand_history.index[-1]
        next_date = last_date + pd.DateOffset(months=1)
        demand_history = pd.concat(
            [demand_history, pd.Series([prediction], index=[next_date])],
            axis=0
        )
    return np.array(predictions), np.array(fallback_flags, dtype=bool)



# ---------------------------
# HYBRID GBM WITH BASE FEATURES
# ---------------------------

def train_hybrid_gbm_base_group(group_train_dfs, extra_features=None, lookback=6, random_state=42):
    # trainiert ein hybrid modell über mehrere zeitreihen (also alle innerhalb der gruppe dann jeweils) mit gbm aber als regressor! nicht "direkt"
    # das dann im rekursiven forecasting genutzt wird
    if extra_features is None:
        extra_features = []

    feature_extractor = create_features_variant_f
    base_forecaster   = croston_forecast

    feature_parts  = []
    residual_parts = []
    # über die trainingsdfs (also alle materialien der entsprechenden gruppe) iterieren
    for series_df in group_train_dfs:
        # features holen
        feature_matrix = feature_extractor(series_df, extra_features)
        # rest wie gewohnt, siehe weiter oben alles als Vgl.
        base_fc_series, _ = recursive_forecast_base(
            base_forecaster,
            series_df["demand"],
            lookback,
            alpha=0.1
        )
        trimmed_features = feature_matrix.iloc[-len(base_fc_series):].copy()
        trimmed_features["baseline_fc"] = base_fc_series
        trimmed_features["residual"]    = (
            series_df["demand"]
            .iloc[-len(base_fc_series):]
            .values
            - base_fc_series
        )
        # inputs und targets separieren
        feature_parts.append(trimmed_features.drop(columns=["demand", "residual"]))
        residual_parts.append(trimmed_features["residual"])

    # reihen zusammenführen
    X_train = pd.concat(feature_parts,  ignore_index=True)
    y_train = pd.concat(residual_parts, ignore_index=True)
    # gbm drauf trainieren
    gbm_model = GradientBoostingRegressor(random_state=random_state)
    gbm_model.fit(X_train, y_train)

    return {
        "model":             gbm_model,
        "feature_extractor": feature_extractor,
        "base_forecaster":   base_forecaster,
        "lag":               lookback,
        "extra_features":    extra_features,
        "feature_names":     X_train.columns.tolist()
    }


def recursive_hybrid_gbm_forecast_pretrained_base(pretrained_model, training_df, forecast_horizon):
    # jetzt rekursiv das hybrid-gbm nutzen zum forecasten
    gbm_model         = pretrained_model["model"]
    feature_extractor = pretrained_model["feature_extractor"]
    base_forecaster   = pretrained_model["base_forecaster"]
    lookback          = pretrained_model["lag"]
    extra_features    = pretrained_model["extra_features"]
    feature_columns   = pretrained_model["feature_names"]

    demand_history = training_df["demand"].copy()
    if len(demand_history) < lookback + 1:
        base_forecast, _ = base_forecaster(
            demand_history, forecast_horizon=forecast_horizon, alpha=0.1
        )
        # hier wieder dasselbe TODO, also ob wir auf Croston zurückfallen bzw. base forecast oder lieber komplett auf 0.
        fallback_flags = np.ones(forecast_horizon, dtype=bool)
        return base_forecast, fallback_flags

    # extra features mitnehmen
    last_extra_values = {
        name: training_df.iloc[-1].get(name, 0)
        for name in extra_features
    }

    predictions    = []
    fallback_flags = []
    # und wieder über den forecast horizon iterieren
    for _ in range(forecast_horizon):
        base_fc_arr, base_was_fallback_arr = base_forecaster(
            demand_history, forecast_horizon=1, alpha=0.1
        )
        base_value        = base_fc_arr[0]
        base_was_fallback = bool(base_was_fallback_arr)

        context_df = demand_history.iloc[-lookback:].to_frame("demand")
        last_date  = demand_history.index[-1]
        next_date  = last_date + pd.DateOffset(months=1)
        augmented_df = pd.concat(
            [context_df, pd.DataFrame({"demand": [base_value]}, index=[next_date])],
            axis=0
        )
        for name in extra_features:
            augmented_df[name] = last_extra_values[name]

        feature_matrix = feature_extractor(augmented_df, extra_features)
        if feature_matrix.empty:
            forecast_value = base_value
            used_fallback  = True
        else:
            feature_matrix = feature_matrix.fillna(0).infer_objects(copy=False)
            feature_matrix.loc[feature_matrix.index[-1], "baseline_fc"] = base_value
            X_new = (
                feature_matrix
                .iloc[[-1]]
                .drop(columns=["demand"])
                .reindex(columns=feature_columns, fill_value=0)
            )
            residual       = gbm_model.predict(X_new)[0]
            forecast_value = max(base_value + residual, 0)
            used_fallback  = False

        predictions.append(forecast_value)
        fallback_flags.append(base_was_fallback or used_fallback)
        last_date = demand_history.index[-1]
        next_date = last_date + pd.DateOffset(months=1)
        demand_history = pd.concat(
            [demand_history, pd.Series([forecast_value], index=[next_date])],
            axis=0
        )

    return np.array(predictions), np.array(fallback_flags, dtype=bool)
