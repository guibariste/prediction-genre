import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib



# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('babynames-clean.csv', header=None, names=['Name', 'Gender'])

# Extraire la dernière lettre de chaque prénom
df['Last_Letter'] = df['Name'].str[-1]

# Convertir 'boy' en 0 et 'girl' en 1
df['Gender'] = df['Gender'].map({'boy': 0, 'girl': 1})

# Conserver uniquement les colonnes nécessaires pour l'entraînement
df = df[['Last_Letter', 'Gender']]


one_hot = pd.get_dummies(df['Last_Letter'])

one_hot = one_hot.astype(int)
# print(one_hot)

# Concaténer les nouvelles colonnes (one-hot) avec la colonne 'Gender'
df = pd.concat([one_hot, df['Gender']], axis=1)


# print(df.head())









X = df.drop('Gender', axis=1)  # Toutes les colonnes sauf 'Gender'
y = df['Gender']  # La colonne 'Gender'

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Définir les paramètres à tester dans GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],       # Nombre d'arbres dans la forêt
    'max_depth': [None, 10, 20, 30],       # Profondeur maximale des arbres
    'min_samples_split': [2, 5, 10],       # Nombre minimum d'échantillons requis pour diviser un nœud
    'min_samples_leaf': [1, 2, 4]          # Nombre minimum d'échantillons requis pour être à une feuille
}

# Configurer GridSearchCV avec le modèle, les paramètres, et la validation croisée
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Exécuter GridSearchCV pour trouver les meilleurs paramètres
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres trouvés par GridSearchCV :")
print(grid_search.best_params_)

# Afficher les scores de chaque combinaison de paramètres testée
results = pd.DataFrame(grid_search.cv_results_)
print("\nScores de chaque combinaison de paramètres testée :")
print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

# Faire des prédictions sur l'ensemble de test avec le meilleur modèle
y_pred = grid_search.best_estimator_.predict(X_test)
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'model1.pkl')
# Évaluer les performances du modèle avec le meilleur ensemble de paramètres
print("\nRapport de classification avec le meilleur modèle :")
print(classification_report(y_test, y_pred))

# Calculer et afficher la précision du meilleur modèle sur l'ensemble de test
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du meilleur modèle sur l'ensemble de test : {accuracy:.4f}")

#Meilleur modele acurracy 77%