#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

    {'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 564348, 'total_payments': 564348, 
    'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'james.prentice@enron.com', 'restricted_stock_deferred': 'NaN', 
    'deferred_income': 'NaN', 'total_stock_value': 1095040, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 
    'exercised_stock_options': 886231, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 
    'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 208809, 'director_fees': 'NaN'}

    LAY KENNETH L
    FASTOW ANDREW S
    
"""

import joblib


enron_data = joblib.load(open("./final_project/final_project_dataset.pkl", "rb"))

null_payments = 0
total_pple = len(enron_data)
poi = 0

for person in enron_data:
    if enron_data[person]['poi'] == True:
        poi += 1
    if enron_data[person]['total_payments'] == 'NaN':
        null_payments += 1
    

print((null_payments / total_pple) * 100)
print(total_pple)
print(null_payments)
print(poi)





