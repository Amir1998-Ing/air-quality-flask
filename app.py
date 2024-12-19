from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle finale sauvegardé
model = joblib.load('linear_regression_model.pkl')

# Route d'accueil (afficher le formulaire)
@app.route('/')
def home():
    return render_template('index.html')

# Route pour voir le contenu du fichier .pkl
@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        # Charger le fichier .pkl
        model_info = joblib.load('linear_regression_model.pkl')

        # Vérifier le type de l'objet chargé
        if hasattr(model_info, 'coef_') and hasattr(model_info, 'intercept_'):
            # Si c'est un modèle de régression linéaire
            return jsonify({
                'type': str(type(model_info)),
                'coefficients': model_info.coef_.tolist(),
                'intercept': model_info.intercept_.tolist()
            })
        elif isinstance(model_info, dict):
            # Si c'est un dictionnaire
            return jsonify({
                'type': 'dict',
                'content': model_info
            })
        else:
            # Autres types de données
            return jsonify({
                'type': str(type(model_info)),
                'content': str(model_info)
            })
    except Exception as e:
        return jsonify({'error': str(e)})

# Route pour faire des prédictions (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4']),
            float(request.form['feature5']),
            float(request.form['feature6']),
            float(request.form['feature7']),
            float(request.form['feature8']),
            float(request.form['feature9']),
            float(request.form['feature10']),
            float(request.form['feature11']),
            float(request.form['feature12'])
        ]
        
        # Convertir les données en tableau numpy
        features_array = np.array(features).reshape(1, -1)

        # Faire la prédiction
        prediction = model.predict(features_array)

        # Retourner la prédiction et afficher dans le même formulaire
        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True)
