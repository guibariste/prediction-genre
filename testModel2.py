import pandas as pd
import joblib
import argparse



def count_vowels(name):
    vowels = "aeiouyAEIOUY"
    return sum(1 for letter in name if letter in vowels)

# Ajouter une colonne qui compte le nombre de voyelles


# Fonction pour compter le nombre de consonnes
def count_consonants(name):
    consonants = "bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ"
    return sum(1 for letter in name if letter in consonants)


def predict_gender(name):


    # Charger le modèle et l'encodeur
    model = joblib.load('model2.pkl')
    one_hot_encoder = joblib.load('oneHotEncoder.pkl')

    # Extraire les lettres nécessaires
    last_letter = name[-1].lower()
    last_2_letters = name[-2:].lower()
    last_3_letters = name[-3:].lower()
    
    # Créer une DataFrame avec les lettres
    letter_df = pd.DataFrame([[last_letter, last_2_letters, last_3_letters]], columns=['Last_Letter', 'Last_2_Letters', 'Last_3_Letters'])
    
    # Encoder les lettres avec OneHotEncoder
    one_hot_encoded = one_hot_encoder.transform(letter_df.astype(str))
    
    # Ajouter des caractéristiques supplémentaires : longueur du nom, nombre de voyelles, nombre de consonnes
    name_length = len(name)
    vowel_count = count_vowels(name)
    consonant_count = count_consonants(name)
    
    # Créer un DataFrame avec ces nouvelles caractéristiques
    extra_features = pd.DataFrame([[name_length, vowel_count, consonant_count]], columns=['Name_Length', 'Vowel_Count', 'Consonant_Count'])
    
    # Concaténer les caractéristiques encodées avec les autres caractéristiques
    final_features = pd.concat([pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out()), extra_features], axis=1)
    
    # Faire la prédiction avec le modèle
    prediction = model.predict(final_features)
    probabilities = model.predict_proba(final_features)

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