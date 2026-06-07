# customer_intelligence.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ─────────────────────────────────────────────────
df = pd.read_csv('Country-data.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ── Basic Info ────────────────────────────────────────────────
print("\n--- Info ---")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistics:\n", df.describe())

# ── Plot 1: Distribution of all numeric features ──────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

df[numeric_cols].hist(figsize=(14, 10), bins=20, color='steelblue', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('plot_distributions.png')
plt.show()

# ── Plot 2: Correlation Heatmap ───────────────────────────────
plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_heatmap.png')
plt.show()

# ── Plot 3: Boxplots for outlier detection ────────────────────
plt.figure(figsize=(14, 6))
df[numeric_cols].boxplot(figsize=(14, 6))
plt.xticks(rotation=45)
plt.title('Boxplots — Outlier Detection')
plt.tight_layout()
plt.savefig('plot_boxplots.png')
plt.show()

# ── Plot 4: Top 10 countries by GDPP ─────────────────────────
top10 = df.nlargest(10, 'gdpp')[['country', 'gdpp']]
plt.figure(figsize=(10, 5))
sns.barplot(data=top10, x='gdpp', y='country', palette='Blues_d')
plt.title('Top 10 Countries by GDP per Capita')
plt.tight_layout()
plt.savefig('plot_top10_gdpp.png')
plt.show()

# ── Feature Engineering ───────────────────────────────────────

# 1. Income-to-GDPP Ratio
df['income_gdpp_ratio'] = df['income'] / (df['gdpp'] + 1)

# 2. Trade Balance (Exports - Imports)
df['trade_balance'] = df['exports'] - df['imports']

# 3. Health Investment Score
df['health_investment'] = df['health'] * df['gdpp'] / 100

# 4. Development Score (composite)
df['dev_score'] = (
    (df['gdpp']       / df['gdpp'].max()) * 0.3 +
    (df['life_expec'] / df['life_expec'].max()) * 0.3 +
    (1 - df['child_mort'] / df['child_mort'].max()) * 0.2 +
    (df['income']     / df['income'].max()) * 0.2
)

# ── Create Classification Target Label ───────────────────────
# Bin countries into 3 development tiers based on dev_score
df['dev_label'] = pd.qcut(df['dev_score'], q=3,
                           labels=['Underdeveloped', 'Developing', 'Developed'])

print("\nLabel Distribution:")
print(df['dev_label'].value_counts())

# ── Plot label distribution ───────────────────────────────────
df['dev_label'].value_counts().plot(kind='bar', color=['#e74c3c','#f39c12','#2ecc71'],
                                     figsize=(7, 4), edgecolor='black')
plt.title('Country Development Level Distribution')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot_label_dist.png')
plt.show()

# ── Prepare Features & Target ─────────────────────────────────
drop_cols = ['country', 'dev_label', 'dev_score']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df['dev_label'].copy()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)   # 0=Developed, 1=Developing, 2=Underdeveloped

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── Helper: Train, Evaluate, Plot Confusion Matrix ────────────
def evaluate_classifier(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    cv    = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  CV Accuracy    : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n{classification_report(y_te, preds, target_names=le.classes_)}")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, preds, display_labels=le.classes_, ax=ax, cmap='Blues'
    )
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(f'plot_cm_{name.replace(" ", "_")}.png')
    plt.show()

    return model, acc

# ── Train All 3 Classifiers ───────────────────────────────────
lr_clf,  lr_acc  = evaluate_classifier("Logistic Regression",
                    LogisticRegression(max_iter=1000, random_state=42),
                    X_train, y_train, X_test, y_test)

rf_clf,  rf_acc  = evaluate_classifier("Random Forest",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    X_train, y_train, X_test, y_test)

xgb_clf, xgb_acc = evaluate_classifier("XGBoost",
                    XGBClassifier(n_estimators=100, random_state=42,
                                  use_label_encoder=False, eval_metric='mlogloss'),
                    X_train, y_train, X_test, y_test)

# ── Model Comparison Bar Chart ────────────────────────────────
models     = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [lr_acc, rf_acc, xgb_acc]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#3498db','#2ecc71','#e74c3c'], edgecolor='black')
plt.ylim(0.5, 1.05)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('plot_model_comparison.png')
plt.show()

# ── Tune Random Forest ────────────────────────────────────────
print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth':    [3, 5, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("RF Best Params   :", rf_grid.best_params_)
print("RF Best CV Acc   :", round(rf_grid.best_score_, 4))

# ── Tune XGBoost ──────────────────────────────────────────────
print("\nTuning XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth':    [3, 5],
    'learning_rate':[0.05, 0.1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    xgb_params, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
print("XGB Best Params  :", xgb_grid.best_params_)
print("XGB Best CV Acc  :", round(xgb_grid.best_score_, 4))

# ── Best Model Final Evaluation ───────────────────────────────
best_clf  = xgb_grid.best_estimator_
best_preds = best_clf.predict(X_test)
print("\n--- Final Best Model (XGBoost Tuned) ---")
print(f"Accuracy: {accuracy_score(y_test, best_preds):.4f}")
print(classification_report(y_test, best_preds, target_names=le.classes_))

# Feature Importance
feat_imp = pd.Series(best_clf.feature_importances_, index=feature_cols)
feat_imp.sort_values().plot(kind='barh', figsize=(9, 6), color='steelblue')
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.savefig('plot_feature_importance.png')
plt.show()

# ── Use only numeric features for clustering ──────────────────
X_cluster = scaler.fit_transform(df[numeric_cols])

# ── Elbow Method to find best K ───────────────────────────────
inertias = []
K_range  = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, marker='o', color='steelblue')
plt.title('Elbow Method — Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
plt.legend()
plt.tight_layout()
plt.savefig('plot_elbow.png')
plt.show()

# ── Apply K-Means with K=3 ────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_cluster)

print("\nK-Means Cluster Distribution:")
print(df['KMeans_Cluster'].value_counts())

# ── Visualize Clusters using PCA (2D) ────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(9, 6))
colors = ['#e74c3c', '#2ecc71', '#3498db']
for cluster in range(3):
    mask = df['KMeans_Cluster'] == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[cluster], label=f'Cluster {cluster}',
                s=80, alpha=0.8, edgecolors='black', linewidths=0.5)
plt.title('K-Means Clusters (PCA 2D View)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('plot_kmeans_pca.png')
plt.show()

# ── Cluster Profile Summary ───────────────────────────────────
cluster_profile = df.groupby('KMeans_Cluster')[numeric_cols].mean().round(2)
print("\nCluster Profiles:")
print(cluster_profile)
cluster_profile.to_csv('kmeans_cluster_profiles.csv')

# ── Find optimal eps using Nearest Neighbors ─────────────────
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_cluster)
distances, _ = neighbors.kneighbors(X_cluster)
distances     = np.sort(distances[:, 4])

plt.figure(figsize=(8, 4))
plt.plot(distances, color='steelblue')
plt.title('K-Distance Graph — Optimal eps for DBSCAN')
plt.xlabel('Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.tight_layout()
plt.savefig('plot_dbscan_eps.png')
plt.show()

# ── Apply DBSCAN ──────────────────────────────────────────────
dbscan = DBSCAN(eps=2.5, min_samples=4)    # adjust eps based on above plot
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_cluster)

n_clusters  = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise     = (df['DBSCAN_Cluster'] == -1).sum()

print(f"\nDBSCAN — Clusters Found : {n_clusters}")
print(f"DBSCAN — Noise Points   : {n_noise}")
print(df['DBSCAN_Cluster'].value_counts())

# ── DBSCAN Visualization ──────────────────────────────────────
plt.figure(figsize=(9, 6))
unique_labels = sorted(df['DBSCAN_Cluster'].unique())
palette = plt.cm.tab10.colors

for label in unique_labels:
    mask  = df['DBSCAN_Cluster'] == label
    color = 'black' if label == -1 else palette[label % 10]
    name  = 'Noise' if label == -1 else f'Cluster {label}'
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=[color], label=name, s=80, alpha=0.8,
                edgecolors='white', linewidths=0.4)

plt.title('DBSCAN Clusters (PCA 2D View)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('plot_dbscan_pca.png')
plt.show()

# ── K-Means Segment Naming ────────────────────────────────────
segment_map = {}
for cluster in range(3):
    avg_gdpp = df[df['KMeans_Cluster'] == cluster]['gdpp'].mean()
    avg_mort = df[df['KMeans_Cluster'] == cluster]['child_mort'].mean()
    if avg_gdpp > 15000:
        segment_map[cluster] = 'High-Value (Developed)'
    elif avg_mort > 50:
        segment_map[cluster] = 'At-Risk (Underdeveloped)'
    else:
        segment_map[cluster] = 'Growth (Developing)'

df['Segment'] = df['KMeans_Cluster'].map(segment_map)

print("\nSegment Distribution:")
print(df['Segment'].value_counts())

# ── Sample countries per segment ─────────────────────────────
for seg in df['Segment'].unique():
    countries = df[df['Segment'] == seg]['country'].tolist()
    print(f"\n{seg} ({len(countries)} countries):")
    print(", ".join(countries[:10]), "...")

# ── Radar chart: Cluster Profile ─────────────────────────────
key_features = ['child_mort', 'income', 'life_expec', 'gdpp', 'health']
cluster_means = df.groupby('KMeans_Cluster')[key_features].mean()

# Normalize to 0-1 for radar
norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors  = ['#e74c3c', '#2ecc71', '#3498db']

for i, (idx, row) in enumerate(norm.iterrows()):
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, 'o-', color=colors[i], linewidth=2,
            label=segment_map.get(idx, f'Cluster {idx}'))
    ax.fill(angles, vals, alpha=0.1, color=colors[i])

ax.set_thetagrids(np.degrees(angles[:-1]), key_features)
ax.set_title('Cluster Radar Chart', size=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('plot_radar.png')
plt.show()

# ── Final output CSV ──────────────────────────────────────────
output = df[['country', 'gdpp', 'income', 'child_mort', 'life_expec',
             'KMeans_Cluster', 'DBSCAN_Cluster', 'Segment', 'dev_label']].copy()

output.to_csv('customer_intelligence_results.csv', index=False)

print("\n✅ All done! Files saved:")
print("   → customer_intelligence_results.csv")
print("   → kmeans_cluster_profiles.csv")
print("   → plot_distributions.png")
print("   → plot_heatmap.png")
print("   → plot_boxplots.png")
print("   → plot_label_dist.png")
print("   → plot_model_comparison.png")
print("   → plot_cm_*.png (3 confusion matrices)")
print("   → plot_feature_importance.png")
print("   → plot_elbow.png")
print("   → plot_kmeans_pca.png")
print("   → plot_dbscan_eps.png")
print("   → plot_dbscan_pca.png")
print("   → plot_radar.png")
