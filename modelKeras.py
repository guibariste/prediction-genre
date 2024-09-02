import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('babynames-clean.csv', header=None, names=['Name', 'Gender'])

# Extraire les lettres
df['Last_Letter'] = df['Name'].str[-1]
df['Last_2_Letters'] = df['Name'].str[-2:]
df['Last_3_Letters'] = df['Name'].str[-3:]

# Convertir 'boy' en 0 et 'girl' en 1
df['Gender'] = df['Gender'].map({'boy': 0, 'girl': 1})

# Ajouter des colonnes pour la longueur du nom, le nombre de voyelles, et le nombre de consonnes
df['Name_Length'] = df['Name'].str.len()

def count_vowels(name):
    vowels = "aeiouyAEIOUY"
    return sum(1 for letter in name if letter in vowels)

df['Vowel_Count'] = df['Name'].apply(count_vowels)

def count_consonants(name):
    consonants = "bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ"
    return sum(1 for letter in name if letter in consonants)

df['Consonant_Count'] = df['Name'].apply(count_consonants)

# Créer un encodeur OneHotEncoder avec handle_unknown='ignore'
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Appliquer OneHotEncoder sur les colonnes catégorielles
X_categorical = one_hot_encoder.fit_transform(df[['Last_Letter', 'Last_2_Letters', 'Last_3_Letters']].astype(str))

# Concaténer les colonnes encodées avec les autres colonnes numériques (Name_Length, Vowel_Count, Consonant_Count)
X_numerical = df[['Name_Length', 'Vowel_Count', 'Consonant_Count']]
X = pd.concat([pd.DataFrame(X_categorical, columns=one_hot_encoder.get_feature_names_out()), X_numerical.reset_index(drop=True)], axis=1)

# Séparer les caractéristiques et la cible
y = df['Gender']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les labels en format one-hot (utile pour la classification binaire dans Keras)
y_train_onehot = to_categorical(y_train, num_classes=2)
y_test_onehot = to_categorical(y_test, num_classes=2)

# Créer le modèle Keras
model = Sequential()

# Ajouter des couches au modèle
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Couche d'entrée avec 64 neurones

model.add(Dense(32, activation='relu'))  # Couche cachée avec 32 neurones
model.add(Dense(2, activation='softmax'))  # Couche de sortie avec 2 neurones (classification binaire)

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train_onehot, epochs=20, batch_size=32, validation_data=(X_test, y_test_onehot))

# Évaluer les performances du modèle
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f"Précision sur l'ensemble de test : {accuracy:.4f}")

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Afficher le rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred_classes))
