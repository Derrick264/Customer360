from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import tempfile

app = Flask(__name__)

# Load the pre-trained machine learning model
filename = 'customer_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the dataset
df = pd.read_csv("Clustered_Customer_Data.csv")

# Define the sorting function (Bubble Sort)
def bubble_sort(data):
    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
    return data

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('new.html')

# Route to handle form submissions and display results
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    balance = float(request.form['balance'])
    balance_frequency = float(request.form['balance_frequency'])
    purchases = float(request.form['purchases'])
    oneoff_purchases = float(request.form['oneoff_purchases'])
    installments_purchases = float(request.form['installments_purchases'])
    cash_advance = float(request.form['cash_advance'])
    purchases_frequency = float(request.form['purchases_frequency'])
    oneoff_purchases_frequency = float(request.form['oneoff_purchases_frequency'])
    purchases_installment_frequency = float(request.form['purchases_installment_frequency'])
    cash_advance_frequency = float(request.form['cash_advance_frequency'])
    cash_advance_trx = float(request.form['cash_advance_trx'])
    purchases_trx = float(request.form['purchases_trx'])
    credit_limit = float(request.form['credit_limit'])
    payments = float(request.form['payments'])
    minimum_payments = float(request.form['minimum_payments'])
    prc_full_payment = float(request.form['prc_full_payment'])
    tenure = float(request.form['tenure'])

    # Prepare data for prediction
    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]
    
    # Use bubble sort on a subset of the data (for example, sorting purchases and cash_advance)
    sorted_data = bubble_sort([purchases, oneoff_purchases, installments_purchases, cash_advance])
    
    # Integrate sorted values back into the data
    data[0][2] = sorted_data[0]  # purchases
    data[0][3] = sorted_data[1]  # oneoff_purchases
    data[0][4] = sorted_data[2]  # installments_purchases
    data[0][5] = sorted_data[3]  # cash_advance

    # Predict cluster
    cluster = loaded_model.predict(data)[0]

    # Pass the predicted cluster to the results template
    return render_template('result.html', cluster=cluster)

if __name__ == '__main__':
    app.run(debug=True)
