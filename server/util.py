import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def predict_estimate_price(location,sqft,bhk,bath):
    load_saved_data()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    load_saved_data()
    return __locations

def load_saved_data():
    print("Loading the saved Data..... Start")
    global __data_columns
    global __locations
    global __model

    with open("../model/columns.json","r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("../model/banglore_model.pickle","rb") as f:
        __model = pickle.load(f)

    print("loading of saved data is completed")

if __name__ == "__main__":
    # load_saved_data()
    print(get_location_name())
    # print(len(get_location_name()))
    print(predict_estimate_price('1st phase jp nagar',1000,3,3))
    print(predict_estimate_price('1st phase jp nagar',1000,2,2))
    print(predict_estimate_price('kambipura',1000,2,2))
    print(predict_estimate_price('indra nagar',1000,2,2))
