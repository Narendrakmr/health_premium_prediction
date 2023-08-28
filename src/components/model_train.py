import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Dataset","trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model = RandomForestRegressor(n_estimators= 200,min_samples_split=5,
                                          min_samples_leaf=2,max_features= 1.0 ,
                                          max_depth=15, n_jobs=-1,random_state=11)
            trained_model=model.fit(X_train,y_train)
            
            predicted_val=trained_model.predict(X_test)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=trained_model
            )

            r2_square = r2_score(y_test,predicted_val)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)