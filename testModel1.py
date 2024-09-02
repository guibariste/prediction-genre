import pandas as pd
import argparse
import joblib

# Charger le modèle pré-entraîné
model = joblib.load('model1.pkl')

# Définir la fonction pour faire des prédictions
def predict_gender(name):
    # Extraire la dernière lettre du prénom
    last_letter = name[-1].lower()
    
    # Créer une DataFrame avec la dernière lettre
    letter_df = pd.DataFrame([[last_letter]], columns=['Last_Letter'])
    
    # Recréer l'encodage one-hot exactement comme lors de l'entraînement
    one_hot = pd.get_dummies(letter_df['Last_Letter'])
    
    
    # Ajouter les colonnes manquantes pour correspondre au modèle
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if letter not in one_hot.columns:
            one_hot[letter] = 0
    
    one_hot = one_hot.sort_index(axis=1)
    # print(one_hot)
    
    # Faire la prédiction avec le modèle
    prediction = model.predict(one_hot)
    probabilities = model.predict_proba(one_hot)

    print(probabilities)
    # Retourner le résultat : 0 pour garçon, 1 pour fille
    return 'girl' if prediction[0] == 1 else 'boy'


#code pour tester un prenom en argument dans le terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict gender based on name.')
    parser.add_argument('name', type=str, help='Name to predict gender for')
    args = parser.parse_args()

    gender = predict_gender(args.name)
    print(f"The predicted gender for the name '{args.name}' is: {gender}")
