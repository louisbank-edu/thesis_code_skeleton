import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data_file = "DATA_PATH/monthly_complete.csv"
output_file = "GROUPING_DIR/grouping_intermittency_similarity.csv"
# n_clusters wird nach elbow-chart manuell angepasst -> 5 sieht gut aus
n_clusters = 5  # anzahl der cluster die wir wollen

df_base = pd.read_csv(data_file, parse_dates=["Buch.dat."])
# ganzzahlingen monatsindex für die lückenberechnung
df = df_base.copy()
df['month_idx'] = df['Buch.dat.'].dt.year * 12 + df['Buch.dat.'].dt.month

# jetzt über alle materialien iterieren und die features berechnen
feature_list = []
for mat, grp in df.groupby('Material'):
    grp = grp.sort_values('month_idx')
    # nonzero werte und "aktiven" bereich finden
    nz = grp.loc[grp['Menge in ErfassME'] > 0]
    if nz.empty:
        continue
    start_idx = nz['month_idx'].iloc[0]
    active = grp.loc[grp['month_idx'] >= start_idx].copy()
    vals = active['Menge in ErfassME'].to_numpy()
    zeros_bool_list = (vals == 0)

    #1: frequenz/lücken features
    p_zero = zeros_bool_list.mean() # anteil der zero values
    # runs zählen und längen bestimmen
    run_lengths = [] #die länge der nullperioden
    count = 0
    for z in zeros_bool_list:
        if z:
            count += 1
        else:
            if count > 0:
                run_lengths.append(count)
                count = 0
    if count > 0:
        run_lengths.append(count)
    run_count = len(run_lengths) # anzahl der lücken
    mean_gap = np.mean(run_lengths) if run_lengths else 0.0 # durchschnittliche länge der lücken
    max_gap = np.max(run_lengths) if run_lengths else 0.0 # maximale länge der lücken


    #2: volume/instentiy features
    nz_vals = vals[vals > 0]
    mean_nz = nz_vals.mean() if nz_vals.size > 0 else 0.0 #durchschnittlicher wert der nonzero werte
    std_nz = nz_vals.std(ddof=0) if nz_vals.size > 0 else 0.0 # standardabweichung der nonzero werte
    max_nz = nz_vals.max() if nz_vals.size > 0 else 0.0 # maximaler wert der nonzero werte
    sum_nz = nz_vals.sum() if nz_vals.size > 0 else 0.0 # summe der nonzero werte



    #3: statistische features
    skew_nz = skew(nz_vals) if nz_vals.size > 1 else 0.0 # schiefe der nonzero werte
    kurt_nz = kurtosis(nz_vals) if nz_vals.size > 1 else 0.0 # kurtosis der nonzero werte


    feature_list.append({
        'Material': mat,
        'p_zero': p_zero,
        'run_count': run_count,
        'mean_gap': mean_gap,
        'max_gap': max_gap,
        'mean_nz': mean_nz,
        'std_nz': std_nz,
        'max_nz': max_nz,
        'sum_nz': sum_nz,
        'skew_nz': skew_nz,
        'kurt_nz': kurt_nz
    })

# features dataframe erstellen
feat_df = pd.DataFrame(feature_list)
feat_df.fillna(0.0, inplace=True)

# die feature matrix skalieren
features = feat_df.drop(columns=['Material'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# elbow-chart zur bestimmung der optimalen cluster-anzahl
inertias = []
k_values = range(1, 11)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow-Chart: Inertia vs. Anzahl Cluster k')
plt.xlabel('Anzahl Cluster k')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
# Hier n_clusters basierend auf Elbow-Chart setzen

# per kmeans clustern
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)
feat_df['GroupID'] = labels + 1

feat_df[['Material', 'GroupID']].to_csv(output_file, index=False)
print(f"Grouping gespeichert nach {output_file}")




# visualisierung
# 2) PCA auf die skalierten Features
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

# 3) Scatterplot zeichnen
plt.figure(figsize=(10, 7))
for gid in sorted(feat_df['GroupID'].unique()):
    mask = feat_df['GroupID'] == gid
    plt.scatter(
        coords[mask, 0],
        coords[mask, 1],
        label=f"Gruppe {gid}",
        s=30,
        alpha=0.6
    )

plt.title('PCA-2D der Intermittency-Features nach KMeans-Gruppierung')
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.legend(loc='best', frameon=False)
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
