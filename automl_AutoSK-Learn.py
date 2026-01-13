import pandas as pd
import autosklearn.regression
from sklearn.model_selection import train_test_split

print("=== Auto-sklearn appliqué au Dataset Housing ===")

# 1. Chargement des données avec Pandas
df = pd.read_csv("Housing.csv")

# 2. Nettoyage
# On retire la colonne 'rownames' pour ne pas nuire au modèle
if 'rownames' in df.columns:
    df = df.drop(columns=['rownames'])

# 3. Elimination des variables prédictibles
y = df['price']
X = df.drop(columns=['price'])

# 4. Split du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Configuration et lancement de l'AutoML
# On définit une limite de temps (ex: 5 minutes = 300s)
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=300,
    per_run_time_limit=30,       # Temps max pour un seul modèle
    memory_limit=3072,           # Limite de RAM en Mo
    seed=42
)

print("\nRecherche du meilleur modèle en cours...")
automl.fit(X_train, y_train)

# 6. Résultats et Leaderboard
print("\n=== Statistiques de l'entraînement ===")
print(automl.sprint_statistics())

# Affichage des modèles retenus dans l'ensemble (Ensemble Leaderboard)
print("\n=== Modèles retenus ===")
print(automl.leaderboard())

automl.fit(X_train, y_train)
# Faire  prédictions
predictions = automl.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nRésultas:")
print(f"Précision: {accuracy:.4f}")
print(f"Meilleur modèle: {automl.show_models()}")
# Classement
print("\nClassement des modèles (Leaderboard):")
print(automl.leaderboard())

print("\n=== Auto-sklearn Terminé ===")