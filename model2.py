import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


#Modele plus complet avec l'ajout du split des 2 dernieres lettres et 3 dernieres lettres (amelioration du modele)
#ainsi que la longueur du mot ,le nombre de voyelles et le nombre de consonnes(peu d'incidence sur l'amelioration )

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('babynames-clean.csv', header=None, names=['Name', 'Gender'])

# Extraire les lettres
df['Last_Letter'] = df['Name'].str[-1]
df['Last_2_Letters'] = df['Name'].str[-2:]
df['Last_3_Letters'] = df['Name'].str[-3:]

# Convertir 'boy' en 0 et 'girl' en 1
df['Gender'] = df['Gender'].map({'boy': 0, 'girl': 1})

# Conserver uniquement les colonnes nécessaires pour l'entraînement
# df = df[['Last_Letter', 'Last_2_Letters', 'Last_3_Letters', 'Gender']]
df['Name_Length'] = df['Name'].str.len()

# Fonction pour compter le nombre de voyelles
def count_vowels(name):
    vowels = "aeiouyAEIOUY"
    return sum(1 for letter in name if letter in vowels)

# Ajouter une colonne qui compte le nombre de voyelles
df['Vowel_Count'] = df['Name'].apply(count_vowels)

# Fonction pour compter le nombre de consonnes
def count_consonants(name):
    consonants = "bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ"
    return sum(1 for letter in name if letter in consonants)

# Ajouter une colonne qui compte le nombre de consonnes
df['Consonant_Count'] = df['Name'].apply(count_consonants)

# print(df.head())

# Créer un encodeur OneHotEncoder avec handle_unknown='ignore'
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


X_categorical = one_hot_encoder.fit_transform(df[['Last_Letter', 'Last_2_Letters', 'Last_3_Letters']].astype(str))

# Concaténer les colonnes encodées avec les autres colonnes numériques (Name_Length, Vowel_Count, Consonant_Count)
X_numerical = df[['Name_Length', 'Vowel_Count', 'Consonant_Count']]
X = pd.concat([pd.DataFrame(X_categorical, columns=one_hot_encoder.get_feature_names_out()), X_numerical.reset_index(drop=True)], axis=1)
# print(X)

# Labels pour l'entrainement
y = df['Gender']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#----------------------------------------------------------------------------------


# model = RandomForestClassifier(random_state=42)

# # Définir les paramètres à tester dans GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],       # Nombre d'arbres dans la forêt
#     'max_depth': [None, 10, 20, 30],       # Profondeur maximale des arbres
#     'min_samples_split': [2, 5, 10],       # Nombre minimum d'échantillons requis pour diviser un nœud
#     'min_samples_leaf': [1, 2, 4]          # Nombre minimum d'échantillons requis pour être à une feuille
# }

# # Configurer GridSearchCV avec le modèle, les paramètres, et la validation croisée
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Exécuter GridSearchCV pour trouver les meilleurs paramètres
# grid_search.fit(X_train, y_train)

# # Afficher les meilleurs paramètres trouvés
# print("Meilleurs paramètres trouvés par GridSearchCV :")
# print(grid_search.best_params_)

# # Afficher les scores de chaque combinaison de paramètres testée
# results = pd.DataFrame(grid_search.cv_results_)
# print("\nScores de chaque combinaison de paramètres testée :")
# print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

# # Faire des prédictions sur l'ensemble de test avec le meilleur modèle
# y_pred = grid_search.best_estimator_.predict(X_test)
# best_model = grid_search.best_estimator_
# joblib.dump(best_model, 'model.pkl')

# # Évaluer les performances du modèle avec le meilleur ensemble de paramètres
# print("\nRapport de classification avec le meilleur modèle :")
# print(classification_report(y_test, y_pred))

# # Calculer et afficher la précision du meilleur modèle sur l'ensemble de test
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Précision du meilleur modèle sur l'ensemble de test : {accuracy:.4f}")

# Définir la fonction pour faire des prédictions

# Meilleurs paramètres trouvés par GridSearchCV :
# {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}



#--------------------------------------------------------------------------------

#meilleur modele avec gridsearch en commentaire ci dessus ,test avec parametres un peu differents mais aucune incidence.


model = RandomForestClassifier(
    n_estimators=300,        # Nombre d'arbres dans la forêt
    max_depth=None,          # Pas de profondeur maximale
    min_samples_split=10,    # Nombre minimum d'échantillons requis pour diviser un nœud
    min_samples_leaf=1,      # Nombre minimum d'échantillons requis pour être à une feuille
    random_state=42          # Pour la reproductibilité
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Sauvegarder le meilleur modèle
joblib.dump(model, 'model2.pkl')
joblib.dump(one_hot_encoder, 'oneHotEncoder.pkl')

# Évaluer les performances du modèle
print("\nRapport de classification avec le modèle ajusté :")
print(classification_report(y_test, y_pred))

# Calculer et afficher la précision du modèle sur l'ensemble de test
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle sur l'ensemble de test : {accuracy:.4f}")


#Meilleur modele acurracy 83%