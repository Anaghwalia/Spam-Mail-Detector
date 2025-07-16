# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:02:52 2025

@author: 91902
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))


#creating a function for predictiomn

def spammailpred(input_data):
    input_data_feature = vectorizer.transform(input_data)

    #making the prediction

    prediction=loaded_model.predict(input_data_feature)

    if(prediction[0] == 1):
      return "Ham Mail"

    else:
      return "Spam Mail"

def main():
    #giving a title for our web page
    
    st.title("Spam Mail Prediction Web App")
    
    
    mail=st.text_input("Enter mail")
    
    pred=''
    
    if st.button('Spam Mail Test'):
        pred=spammailpred([mail])
        
    st.success(pred)


if __name__ == '__main__':
    main()
    
    