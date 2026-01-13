
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.metrics import r2_score, mean_squared_error

print("=== AutoML MLJAR appliqué au Dataset Housing ===")

# 1. Chargement des données
df = pd.read_csv("Housing.csv")

# 2. Nettoyage
if 'rownames' in df.columns:
    df = df.drop(columns=['rownames'])

# 3. Préparation des variables
y = df['price']
X = df.drop(columns=['price'])

# 4. Split du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Configuration de MLJAR
automl = AutoML(
    mode="Explain",
    total_time_limit=300, # 5 minutes
    results_path="AutoML_Housing_Results" # Dossier où seront créés les rapports
)

print("\nDébut de l'entraînement avec MLJAR...")
automl.fit(X_train, y_train)

# 6. Évaluation
predictions = automl.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"\n=== Résultats finaux ===")
print(f"R² Score : {r2:.4f}")
print(f"RMSE : {rmse:.2f}")

# Information
print(f"\nSuccès ! Consultez le dossier 'AutoML_Housing_Results' pour voir les rapports HTML détaillés.")