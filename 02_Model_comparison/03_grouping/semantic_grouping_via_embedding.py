import pandas as pd
import openai
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

openai.api_key = "geheimer_key"
data_file = "DATA_PATH/monthly_complete.csv"
output_file = "GROUPING_DIR/grouping_semantic_auto_named.csv"


df_full = pd.read_csv(data_file)
unique_texts = df_full['Materialkurztext'].dropna().unique().tolist()

#embeddings erzeugen
resp = openai.embeddings.create(
    model="text-embedding-3-small",
    input=unique_texts
)
embeddings = [item.embedding for item in resp.data]

#elbow und slhouette berechnung für k = 2 bis 15
Ks = list(range(2, 16))
inertias = []
silhouettes = []

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(embeddings, km.labels_))

# 4) elbow und silhouette plotten (man sieht keinen so richtig klassischen elbow, aber so 9,10 scheint am besten zu sein)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(Ks, inertias, marker='o')
plt.title("Elbow‑Plot (Inertia vs. k)")
plt.xlabel("Anzahl Cluster k")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(Ks, silhouettes, marker='o')
plt.title("Silhouette‑Score vs. k")
plt.xlabel("Anzahl Cluster k")
plt.ylabel("Silhouette‑Score")

plt.tight_layout()
plt.show()

#best_k automatisch als derjenige mit dem höchsten silhouettescore
best_k = Ks[silhouettes.index(max(silhouettes))]
print(f"Gewähltes best_k = {best_k}")

#KMeans‑Clustering mit best_k
km_final = KMeans(n_clusters=best_k, random_state=42).fit(embeddings)
labels = km_final.labels_

#cluster naming
def name_cluster_with_gpt(samples, idx):
    prompt = f"""Du bist ein prägnanter Kategorisierer.
Gib mir in wenigen Worten einen treffenden, sprechenden Namen für die folgende Gruppe (Cluster #{idx}):

{chr(10).join(f"- {s}" for s in samples)}

Antwort _nur_ mit dem Label."""
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    return resp.choices[0].message.content.strip()

df_cluster = pd.DataFrame({
    'text': unique_texts,
    'label': labels
})

cluster_names = {}
for k in sorted(df_cluster['label'].unique()):
    samples = df_cluster[df_cluster['label']==k]['text'].head(5).tolist()
    cluster_names[k] = name_cluster_with_gpt(samples, k)

print("Automatisch generierte Cluster‑Labels:")
for k, name in cluster_names.items():
    print(f"  Cluster {k}: {name}")

df_full['GroupID'] = df_full['Materialkurztext'] \
    .map({txt: cluster_names[lab] for txt, lab in zip(unique_texts, labels)}) \
    .fillna("Unbekannt / Leer")

out = df_full[['Material', 'GroupID']].drop_duplicates()
out.to_csv(output_file, index=False)
print(f"Grouping CSV gespeichert unter: {output_file}")