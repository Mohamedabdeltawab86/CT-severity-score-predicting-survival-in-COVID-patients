from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    age = request.args.get('age')
    gender = request.args.get("is_male", 1)
    ggo = request.args.get('is_GGO', 1)
    consolidation = request.args.get('is_consolidation', 1)
    crazy_paving = request.args.get('is_crazy_paving', 1)
    CT_severity = request.args.get("CT_severity")
    survival = model.predict(scaler.transform([[age, gender, ggo, consolidation, crazy_paving, CT_severity]]))[0]
    return render_template("index.html", survival=survival)


if __name__ == "__main__":
    app.run()
    