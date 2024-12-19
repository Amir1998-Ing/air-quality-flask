import requests

# URL de l'API
url = "http://127.0.0.1:5000/predict"

# Exemple de données d'entrée
data = {
    "features": [56,14649,0.27434748187244284,1,-0.1251014168798983,0,0.29817249892322484,0,0.06357472619964469,0,0.36520372056338224,1]  # Remplacez par vos propres valeurs
}

# Envoyer une requête POST
response = requests.post(url, json=data)

# Afficher la réponse de l'API
print("Réponse de l'API :", response.json())
