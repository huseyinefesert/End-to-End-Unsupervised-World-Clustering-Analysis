# End-to-End-Unsupervised-World-Clustering-Analysis

This project applies multiple **unsupervised learning** algorithms to segment countries based on key socioeconomic indicators.  
Interactive world maps and visualizations are produced to help understand which countries are similar in terms of health, economy, and development needs.

---

## 🌍 Overview

- **Goal:** Discover and visualize natural groupings of countries according to metrics like child mortality, GDP, health expenditure, and more.
- **Data:** `29-country_data.csv` with 9 features per country (excluding country name).

---

## 🚀 Methods Used

1. **Data Preprocessing**
    - Scaling all features using MinMaxScaler.
    - Dimensionality reduction to 3 principal components via PCA.
2. **Clustering Algorithms**
    - **KMeans (n=3)**
    - **Agglomerative Clustering (n=3)**
    - **DBSCAN** (automatic cluster/outlier discovery)
    - **HDBSCAN** (automatic cluster/outlier discovery)
3. **Cluster Evaluation**
    - Each method evaluated using **Silhouette Score**.
    - Visualized with boxplots (`child_mort` vs. cluster).
4. **Interactive Mapping**
    - Clusters visualized on world maps with **Plotly**.  
      Each algorithm outputs a color-coded map (`map.html`, `KMeans_map.html`, etc.).

---

## 📊 Results

| Algorithm       | Silhouette Score | 
|-----------------|:---------------:|
| KMeans          | 0.4386          |
| Agglomerative   | 0.3983          |
| DBSCAN          | 0.5992          |
| HDBSCAN         | 0.1230          |

- **Interpretation:**  
  - **Higher Silhouette Score = Better cluster separation and quality.**  
  - In this project, **DBSCAN** provided the highest score, indicating the best natural clustering under these settings.

---

## 🗺️ Example Output

> Below: Example of a color-coded map showing clusters for each algorithm (open HTML files for interactive map).

![Cluster Map Example](https://i.imgur.com/3yBU4pU.png) <!-- Kendi haritanın ekran görüntüsünü buraya yükleyebilirsin -->

---

## 🛠️ Technologies & Libraries

- **Python 3.8+**
- pandas, numpy, scikit-learn
- plotly, seaborn, matplotlib
- hdbscan

---

## 📦 How to Run

1. Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn plotly matplotlib seaborn hdbscan
    ```
2. Place `29-country_data.csv` in the project directory.
3. Run:
    ```bash
    python your_script.py
    ```
4. Open generated HTML files (e.g., `KMeans_map.html`) in your browser to explore interactive maps.

---

## 📚 Notes

- Each algorithm may find a different number and type of clusters.
- Silhouette score, cluster size, and interpretation may change with parameter tuning.
- Maps are automatically saved and opened for user exploration.

---

## ✨ Author

This project was created as an unsupervised machine learning portfolio work for international socioeconomic analysis.

---

**Feel free to fork, adapt, or contribute!**
