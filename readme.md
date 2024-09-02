# prediction genre

## **Prérequis**
Avant de pouvoir exécuter ce projet, vous devez installer Python 3 sur votre système. Suivez les instructions ci-dessous pour installer Python 3 et configurer l'environnement virtuel.

## **Installation de Python 3**
**Linux :**

```bash
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip
```

**MacOs:**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
**Une fois brew installe on installe python 3:**
```
brew install python3
```
**Windows**
Téléchargez l'installateur Python 3 depuis le site officiel : python.org
Lors de l'installation, assurez-vous de cocher l'option "Add Python to PATH".
Excecutez le fichier d'installation.


## **Création et activation de l'environnement virtuel**
Il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances de votre projet.

**Linux et macOS:**
Créez un environnement virtuel :
```bash
python3 -m venv env
```

## **Activez l'environnement virtuel :**
```bash
source env/bin/activate
```
**Windows**
Créez un environnement virtuel :
```bash
python -m venv env
```
**Activez l'environnement virtuel :**
```bash
.\env\Scripts\activate
```


## **Installation des dépendances**
Une fois l'environnement virtuel activé, vous pouvez installer les dépendances nécessaires:
```bash
pip install requests bs4 pandas
```
Mettre le fichier python dans le repertoire de l'environnement virtuel.


## **Exécution du script**
Après avoir installé toutes les dépendances, vous pouvez exécuter le script en utilisant la commande suivante :
```bash
python scrapping.py
```
