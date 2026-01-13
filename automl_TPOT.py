import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=== AutoML avec TPOT appliqué au Dataset Housing ===")

# 1. Chargement des données
df = pd.read_csv("Housing.csv")

# 2. Nettoyage
if 'rownames' in df.columns:
    df = df.drop(columns=['rownames'])

# Pour s'assurer que les données soitent purement numériques
# Convertir  les variables textuelles (yes/no -> 1/0)
cols_to_encode = ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea']
for col in cols_to_encode:
    df[col] = df[col].map({'yes': 1, 'no': 0})

y = df['price']
X = df.drop(columns=['price'])

# 3. Split du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Configuration de TPOT
# generations: nombre d'itérations pour optimiser (plus c'est haut, c'est meilleur )
# population_size: nombre de programmes conservés à chaque génération
tpot = TPOTRegressor(
    generations=5,
    population_size=20,
    verbosity=2,
    random_state=42,
    max_time_mins=5  # On limite à 5 minutes
)

print("\nRecherche du meilleur modèle...")
tpot.fit(X_train, y_train)

# 5. Évaluation
print("\n=== Score sur le jeu de test ===")
score = tpot.score(X_test, y_test) # Retourne le R² par défaut
predictions = tpot.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"R² Score : {score:.4f}")
print(f"RMSE : {rmse:.2f}")

# 6. Exportation du code généré
tpot.export('meilleur_code_housing.py')
print("\nLe code du meilleur modèle a été exporté dans 'meilleur_code_housing.py'")