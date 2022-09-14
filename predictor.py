import joblib
import numpy as np
import tensorflow
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

malaria_model   = load_model("models/malaria_model.h5")
pneumonia_model = load_model("models/pneumonia_model.h5")
diabetes_model  = joblib.load("models/diabetes_model")
cancer_model    = joblib.load("models/cancer_model")
kidney_model    = joblib.load("models/kidney_model")
liver_model     = joblib.load("models/liver_model")
heart_model     = joblib.load("models/heart_model")

def diabetes_predictor(values_list):
    values_list = np.array(values_list).reshape(1,8)
    result = diabetes_model.predict(values_list)
    prediction = ''
    if(int(result[0])==1):
        prediction = 'Sorry ! Suffering from diabetes'
    else:
        prediction = 'Congrats ! you are Healthy <diabetes_predictor>'
    return prediction

def cancer_predictor(values_list):
    values_list = np.array(values_list).reshape(1,30)
    result = cancer_model.predict(values_list)
    prediction = ''
    if(int(result[0])==1):
        prediction = 'Sorry ! Suffering from cancer'
    else:
        prediction = 'Congrats ! you are Healthy <cancer_predictor>'
    return prediction

def kidney_predictor(values_list):
    values_list = np.array(values_list).reshape(1,10)
    result = kidney_model.predict(values_list)
    prediction = ''
    if(int(result[0])==1):
        prediction = 'Sorry ! Suffering from kidney_model'
    else:
        prediction = 'Congrats ! you are Healthy <kidney_predictor>'
    return prediction


def liver_predictor(values_list):
    values_list = np.array(values_list).reshape(1,12)
    result = liver_model.predict(values_list)
    prediction = ''
    if(int(result[0])==1):
        prediction = 'Sorry ! Suffering from Diabetes'
    else:
        prediction = 'Congrats ! you are Healthy <liver_predictor>'
    return prediction

def heart_predictor(values_list):
    values_list = np.array(values_list).reshape(1,11)
    result = heart_model.predict(values_list)
    prediction = ''
    if(int(result[0])==1):
        prediction = 'Sorry ! Suffering from Diabetes'
    else:
        prediction = 'Congrats ! you are Healthy <heart_predictor>'
    return prediction

def malaria_predictor(file_path):
    data = image.load_img(file_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    result = malaria_model.predict(data)
    indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
    print(result)
    predicted_class = np.argmax(result, axis=1)
    accuracy = np.round(result[0][predicted_class] * 100, 2)
    label = indices[predicted_class[0]]
    return label,'\n',accuracy
# D:\Gradution Project\project\ML-MT\uploads\test.png
    """
    indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
    result = predicted
    print(result)
    x = 
    #predicted_class = result.astype(int)
    #np.argmax(result,axis=1)
    accuracy = np.round(result[0][predicted_class] * 100, 2)
    label = indices[predicted_class]
    return label,'\n',accuracy
    #####
    indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
    print(result)
    predicted_class = np.asscalar(np.argmax(result, axis=1))
    accuracy = round(result[0][predicted_class] * 100, 2)
    label = indices[predicted_class]
    
    """

def pneumonia_predictor(file_path):
    data = image.load_img(file_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = pneumonia_model.predict(data)
    indices = {0: 'Normal', 1: 'Pneumonia'}
    result = predicted

    if(result>50):
        label= indices[1]
        accuracy= result
    else:
        label= indices[0]
        accuracy= 100-result
    return label
    return accuracy
    
if __name__=="__main__":
    print("""TheClinic App predictor [listed test numbers ]\n
          1_ diabetes  test\n
          2_ cancer    test\n
          3_ kidney    test\n
          4_ liver     test\n
          5_ heart     test\n
          6_ malaria   test\n
          7_ pneumonia test
          """)
    choose = int(input("Enter Test Number : "))
    if  (choose == 1):
        values = list(map(float,input("Enter the numbers : ").strip().split()))
        print(diabetes_predictor(values))
        
    elif(choose == 2):
        values = list(map(float,input("Enter the numbers : ").strip().split()))
        print(cancer_predictor(values))
        
    elif(choose == 3):
        values = list(map(float,input("Enter the numbers : ").strip().split()))
        print(kidney_predictor(values))
        
    elif(choose == 4):
        values = list(map(float,input("Enter the numbers : ").strip().split()))
        print(liver_predictor(values))
        
    elif(choose == 5):
        values = list(map(float,input("Enter the numbers : ").strip().split()))
        print(heart_predictor(values))
        
    elif(choose == 6):
        file_path = input("Enter the file path : ")
        print(malaria_predictor(file_path))
    elif(choose == 7):
        file_path = input("Enter the file path : ")
        print(pneumonia_predictor(file_path))
    else:
        print('Error ! sorry .')
        
