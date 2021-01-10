from flask import Flask, render_template, request
import joblib,pickle
import numpy as np

app = Flask(__name__)
# Load the Random Forest model
model = joblib.load('Fraud_deploy.pkl' )


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    payment_method = (request.form['payment_method'])
    partner_id = (request.form['partner_id'])
    partner_category = (request.form['partner_category'])
    device_type = (request.form['device_type'])
    money_transacted = (request.form['money_transacted'])
    partner_pricing_category = (request.form['partner_pricing_category'])
    Hours = (request.form['Hours'])
    day = (request.form['day'])


    data = [[payment_method, partner_id, partner_category, device_type, money_transacted, partner_pricing_category,Hours, day]]
    prediction = model.predict(data)
    if (money_transacted == 0):
        prediction = [0]
        return render_template('index.html', prediction_text='No Transaction')

    else:
        prediction = model.predict(data)
        if prediction == 0:
            return render_template('index.html', prediction_text='Fraud Transaction')

        else:
            return render_template('index.html', prediction_text=' Not Fraud Transaction')


if __name__ == "__main__":
    app.run(debug=True)