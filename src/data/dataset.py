from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, 
                 data: pd.DataFrame, 
                 features: List[str], 
                 target: List[str],
                 random_state: int = 42,
                 test_size: float = 0.2) -> None:
        
        self.features = features
        self.target = target
        self.data = data

        self._test_size = test_size
        self._random_state = random_state

        self._X = data[features]
        self._y = data[target]
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y, test_size=test_size, random_state=random_state)

    @property
    def training(self):
        return self._X_train, self._y_train
    
    @property
    def testing(self):
        return self._X_test, self._y_test
    

    def get_validation_data(self, test_size: float = 0.2):
        validation_data = pd.concat([self._X_test, self._y_test])
        return Dataset(
            data=validation_data,
            features=self.features,
            target=self.target,
            random_state=self._random_state,
            test_size=test_size
        )