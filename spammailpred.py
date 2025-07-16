# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model

loaded_model = pickle.load(open('C:/projects/ml/spammail/trained_model.sav', 'rb'))
vectorizer = pickle.load(open('C:/projects/ml/spammail/vectorizer.sav', 'rb'))
#Building a predictive system

user_input = ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"]

# Convert the input text to a feature vector
input_data_feature = vectorizer.transform(user_input)

#making the prediction

prediction=loaded_model.predict(input_data_feature)

if(prediction[0] == 1):
  print("Ham Mail")

else:
  print("Spam Mail")