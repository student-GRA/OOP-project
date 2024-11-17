import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder




# Parent class. Contains certain methods to be inherited by other classes.
class GLM:
    
    ## Constructs beta values for the instance. When it is initialized it has not been fitted, so i assign betas = 0
    #
    def __init__(self):
         self._betas = 0

    ## Calculates the negative log likelihood. A private class
    # @param params: the beta values to use in the log likelihood equation
    # @param x: the exogenous variables.
    # @param y: the endogenous variables.
    def _negllik(self, params, x, y):
        raise NotImplementedError
    
    ## Fits the model. Calculates the betas that minimize the negative log likelihood.
    # @param x: the exogenous variables.
    # @param y: the endogenous variables. 
    def fit(self, x, y):
        num_samples, num_features = x.shape
        init_params = np.repeat(0.1,num_features)
        results = minimize(self._negllik , init_params , args=(x,y))
        self._betas = results['x']
        return self._fitted_model(self._betas)
    
    
    ## A inner class returned from the "fit" method. A private class that should not be called from outside the class
    # @params betas: The beta values gained in "fit" method which was aquired through minimizing the negative log likelihood.
    class _fitted_model:
        def __init__(self,betas):
            self._betas = betas
            self
        
        ## Makes predictions based on exogenous variables x
        # @params x: exogenous variable.  
        def predict(self, x):
            pred = np.matmul(x, self._betas)
            return pred
        
        ## Returns the fitted beta values
        @property
        def betas(self):
            return self._betas

  
  
  
  
  
  # Child class. Inherits methods from the GLM class
class Normal(GLM):
    ## Constructor. Inherits from its parent class.
    #
    def __init__(self):
        super().__init__()
   
    ## calculates the negative log likelihood for a Normal link function. A private method that should not be accesed from outside the class.
    # @param params: the beta values to use in the log likelihood equation
    # @param x: the exogenous variables.
    # @param y: the endogenous variables.
    def _negllik(self, params, x, y):
        eta = np.matmul(x, params)
        mu = eta
        llik = np.sum(norm.logpdf(y,mu))
        return -llik
    
    ## Fits the model. Calculates the betas that minimize the negative log likelihood. Returns a class containing these beta values. Inherited from the GLM class
    # @param x: the exogenous variables.
    # @param y: the endogenous variables. 
    def fit(self, x, y):
        return super().fit(x, y)


# Child class. Inherits methods from the GLM class.
class Poisson(GLM):
    ## Constructor. Inherits from its parent class.
    #
    def __init__(self):
        super().__init__()
    
    ## calculates the negative log likelihood with a Poisson link function. A Private method.
    # @param params: the beta values to use in the log likelihood equation
    # @param x: the exogenous variables.
    # @param y: the endogenous variables.
    def _negllik(self, params, x, y):
        eta = np.matmul(x, params)
        mu = np.exp(eta)
        llik = np.sum(-mu + y * eta)
        return -llik
    
    ## Fits the model. Calculates the betas that minimize the negative log likelihood. Returns a class containing these beta values. Inherited from the GLM class
    # @param x: the exogenous variables.
    # @param y: the endogenous variables.
    def fit(self, x, y):
        return super().fit(x, y)
    
    ## A private class not to be accesed from outside the class. The  class is returned from the fit method with the fitted beta values.
    # @params betas: The fitted beta values used to make predicitons. The beta values that minimized the negative log likelihood. 
    class _fitted_model(GLM._fitted_model):
        ## Constructor. Inherits from its parent class.
        #
        def __init__(self,betas):
            super().__init__(betas)
        
        ## Makes predictions based on inputted exogenous variable x. Predicts by the use of the beta values gained from the fit method. 
        # @param x: the exogenous variables.
        def predict(self, x):
            pred = super().predict(x) # inherit
            return np.exp(pred) # Override.
    

class Bernoulli(GLM):
    ## Constructor. Inherits from its parent class.
    #
    def __init__(self):
        super().__init__()
    
    ## calculates the negative log likelihood with a Bernoulli link function. 
    # @param params: the beta values to use in the log likelihood equation.
    # @param x: the exogenous variables.
    # @param y: the endogenous variables.
    def _negllik(self, params, x, y):
        eta = np.matmul(x, params)
        llik = np.sum(y*eta - np.log(1+ np.exp(eta)))
        return -llik
    
    ## Fits the model. Calculates the betas that minimize the negative log likelihood. 
    # #Returns a class containing these beta values. Inherited from the GLM class.
    # @param x: the exogenous variables.
    # @param y: the endogenous variables.
    def fit(self, x, y):
        return super().fit(x, y)
    
    ## A private class not to be accesed from outside the class. The  class is returned from the fit method with the fitted beta values.
    # @params betas: The fitted beta values used to make predicitons. The beta values that minimized the negative log likelihood. 
    class _fitted_model(GLM._fitted_model):
        def __init__(self,betas):
            super().__init__(betas)
            
        ## Predicts based on the given endogenous variables x.
        # @params x: exogenous variable used to predict exogenous variable. 
        def predict(self, x):
            pred = super().predict(x) 
            return (np.exp(pred))/(1 + np.exp(pred))


 
 
 