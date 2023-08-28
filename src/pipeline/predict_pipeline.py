import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
class PredictPipeline:
    def __init__(self):
        pass ##this is empty constructor

    def predict(self,df):
        try:
            model_path=os.path.join("Dataset","trained_model.pkl")
            preprocessor_path=os.path.join("Dataset","preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(df)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,age: int,gender :str, bmi: float,children :str,smoker : str,region :str,
                 medical_history : str,family_medical_history : str,exercise_frequency : str,
                 occupation : str,coverage_level :str):

        self.age = age 
        self.gender = gender
        self.bmi = bmi
        self.children=children
        self.smoker =smoker
        self.region = region
        self.medical_history = medical_history
        self.family_medical_history = family_medical_history
        self.exercise_frequency = exercise_frequency
        self.occupation = occupation
        self.coverage_level = coverage_level
        
    def get_data_as_data_frame(self):
        '''this function will return the dataframe of the input given by the user'''
        try:
            custom_data_input_dict = {
                "age" : [self.age],
                "gender": [self.gender],
                "bmi": [self.bmi],
                "children" : [self.children],
                "smoker": [self.smoker],
                "region" : [self.region],
                "medical_history":[self.medical_history],
                "family_medical_history":[self.family_medical_history],
                "exercise_frequency" : [self.exercise_frequency],
                "occupation" : [self.occupation],
                "coverage_level" : [self.coverage_level]
                }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
