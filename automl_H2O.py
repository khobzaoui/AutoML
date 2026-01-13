import h2o
from h2o.automl import H2OAutoML

print("=== H2O AutoML appliqué au Dataset Housing ===")

# 1. Chargement des données
df_path = "Housing.csv"
h2o.init()

# Importation dans H2O
data = h2o.import_file(df_path)

# 2. Nettoyage
# On peut retirer la colonne 'rownames' pour ne pas nuire au modèle
if 'rownames' in data.columns:
    data = data.drop('rownames')

# 3. Suppression de la variable à prédire, ici c'est "Price"
y = "price"
x = data.columns
x.remove(y)

# 4. Split du dataset (Entraînement 80% / Test 20%)
train, test = data.split_frame(ratios=[0.8], seed=42)

print(f"Lignes d'entraînement : {train.shape[0]}")
print(f"Lignes de test : {test.shape[0]}")

# 5. Configuration et lancement de l'AutoML
# On limite à 10 modèles ou 5 minutes
aml = H2OAutoML(
    max_models=10,
    seed=42,
    max_runtime_secs=300,
    verbosity="info" # Afficher le progres
)

print("\nEntraînement en cours...")
aml.train(x=x, y=y, training_frame=train)

# 6. Affichage du Leaderboard
print("\n=== Classement des modèles (Leaderboard) ===")
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

# Step 7: Sélection du meilleur Modèle
print(f"\nMeilleur Modèle: {aml.leader.model_id}")
best_model = aml.leader

# Step 8: Make predictions
print("\n6. Faire les prédictions...")
predictions = best_model.predict(test)
print("Prédictions sample:")
print(predictions.head())

# Step 9: Performance du Modèle
print("\n7. Performance du Modèle:")
performance = best_model.model_performance(test)
print(performance)

# Step 10: Stopper H2O
print("\n8. Arrêt du H2O...")
h2o.cluster().shutdown()

print("\n=== AutoML Terminé ===")