import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


## Parent class containing all methods that can be shared
#
class loader:
    ## Contructor initializing variables that are needed across methods.
    #
    def __init__(self):
        self._data = None
        self._X = None
        self._y = None
        self._endog = None
    
    ## Abstract method. Forces child classes to program it.
    #      
    def load_data(self, endogenous_column):
        raise NotImplementedError
    
    ## Creates a x dataset and a y dataset based on what exogenous variables are wanted.
    # @params exogenous_columns: column names of the columns to be used as predictors.
    def assign_x_y(self, exogenous_columns):
        self._y = self._data.loc[:, self._endog]
        
        if exogenous_columns == 'all':
            self._X = self._data.drop(self._endog, axis = 1)
        else:
            self._X = self._data.loc[:, exogenous_columns]
        
            
    @property
    def dataset(self):
        return self._data
    
    @property
    def x(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    ## Transposes the x matrix
    #
    def x_transpose(self):
        assert isinstance(self._X, pd.DataFrame), 'Please assign endogenous and exogenous variables first'
        self._X = np.transpose(self._X)
    
    ## Adds a intercept to the data. Adds a whole row of ones.
    def add_intercept(self):
        assert isinstance(self._X, pd.DataFrame), 'Please assign endogenous and exogenous variables first'
        ones = [1] * len(self._X)
        assert self._X.iloc[:, 0].equals(pd.Series(ones)) == False, 'An intercept column has already been added.'
        self._X.insert(loc = 0, column='intercept', value = ones)
    
    ## encodes labels for columns of type "object" to enable matrix multiplication.
    #
    def _encode_labels(self):
        idx = [True if el == 'object' else False for el in self._data.dtypes]
        cols = self._data.columns[idx]
        le = LabelEncoder()
        for col in cols:
            self._data[col] = le.fit_transform(self._data[col])
        
        


class sm_loader(loader):
    def __init__(self):
        super().__init__()
    
    ## Loades the statsmodels dataet.
    # @params: title: the title of the dataset to load. Only works for ['duncan', 'spector']
    def load_data(self, title):
        if title == 'duncan':
            loaded = sm.datasets.get_rdataset('Duncan', 'carData')
            self._endog = 'income'
        elif title == 'spector':
            loaded = sm.datasets.spector.load_pandas()
            self._endog = loaded.endog.name
        
        # assign the data to self and encodes the "object" columns.
        self._data = loaded.data
        self._encode_labels()
        
        



class csv_loader(loader):
    def __init__(self):
        super().__init__()
        
    ## Loads the csv data. Only works fully for the warpbreaks dataset for the moment.
    # @params path: the path to the csv file, either local path or http path.
    def load_data(self, path):
        self._data = pd.read_csv(path)
        self._endog = 'breaks'
        self._encode_labels()


