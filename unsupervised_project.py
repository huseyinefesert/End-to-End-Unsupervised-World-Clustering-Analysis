import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import hdbscan
import webbrowser

# --- DATA ---
df = pd.read_csv("country_data.csv")
df2 = df.drop("country", axis=1)

# --- SCALING ---
scaler = MinMaxScaler()
df2 = scaler.fit_transform(df2)
df2 = pd.DataFrame(df2, columns=['child_mort', 'exports', 'health', 'imports', 'income',
                                'inflation', 'life_expec', 'total_fer', 'gdpp'])

# --- PCA ---
pca = PCA()
pca_df2 = pd.DataFrame(pca.fit_transform(df2))
explained = np.cumsum(pca.explained_variance_ratio_)
plt.plot(list(range(1, len(explained) + 1)), explained)
plt.ylabel("Variance Covered")
plt.xlabel("Num Components")
plt.title("Variance Covered by PCA")
plt.show()

pca_df2 = pca_df2.iloc[:, :3]

# --- CLUSTERING & VIS ---
results = []
cluster_methods = [
    ('KMeans', KMeans(n_clusters=3, random_state=42)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=3)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=3 )),
    ('HDBSCAN', hdbscan.HDBSCAN(min_cluster_size=3))
]


# --- CLUSTERING ---
for name, method in cluster_methods:
    if name == 'DBSCAN':
        labels = method.fit_predict(pca_df2)

        try:
            score = silhouette_score(pca_df2, labels) if len(set(labels)) > 1 and (labels != -1).sum() > 0 else np.nan
        except:
            score = np.nan
    else:
        labels = method.fit_predict(pca_df2)
        score = silhouette_score(pca_df2, labels) if len(set(labels)) > 1 else np.nan


    tmp_df = df.copy()
    tmp_df['Cluster'] = labels


    cluster_means = tmp_df.groupby('Cluster')['child_mort'].mean().sort_values()
    name_map = {}

    if len(cluster_means) >= 3:
        idx = cluster_means.index.tolist()
        name_map = {idx[0]: "No Budget Needed", idx[1]: "In Between", idx[2]: "Budget Needed"}
    else:

        name_map = {k: f"Cluster {k}" for k in cluster_means.index}
    tmp_df['Cluster_Name'] = tmp_df['Cluster'].map(name_map)

    results.append((name, score, tmp_df))


    fig = px.choropleth(
        tmp_df,
        locations="country",
        locationmode="country names",
        color="Cluster_Name",
        title=f"{name}: Needed Budget by Country<br>Silhouette Score: {score:.3f}",
        color_discrete_map={
            "Budget Needed": "red",
            "In Between": "yellow",
            "No Budget Needed": "green"
        }
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.write_html(f"{name}_map.html")
    webbrowser.open(f"{name}_map.html")


print("\n--- Silhouette Scores ---")
for name, score, _ in results:
    print(f"{name}: {score:.4f}")


for name, score, tmp_df in results:
    plt.figure(figsize=(7,4))
    sns.boxplot(data=tmp_df, x="Cluster_Name", y="child_mort")
    plt.title(f"{name} - child_mort vs Cluster\nSilhouette: {score:.3f}")
    plt.show()
