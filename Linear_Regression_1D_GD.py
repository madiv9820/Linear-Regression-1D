import numpy as np

'''
    Writing a Linear Regression for one variable from Scratch.
    But using Gradient Descent
'''
class Linear_Regression:
    # Constructor
    def __init__(self, 
                    niterations: int = 1000,
                    alpha: int = 1,
                    stoppage: float = 0.001):
        '''
            niterations: No of iterations you want to run to train the algorithm through gradient descent
            alpha: learning rate, the rate at which intercept and coef will change its value
            stoppage: diff in the cost value for which you want to stop training 
            intercept_: is the y intercept to be predicted (C)
            coef_: is the slope to be predicted (m)
        '''
        self.__niterations = niterations
        self.__alpha = alpha
        self.__stoppage = stoppage
        self.intercept_ = 0
        self.coef_ = 0

    def __repr__(self) -> str:
        return ("class Linear_Regression_1D_GD.Linear_Regression" + 
                f"<niterations = {self.__niterations}, alpha = {self.__alpha}, stoppage = {self.__stoppage}>")


    # Calculate the average cost or difference between the predicted and actual values 
    def __cost(self, 
                    x: np.ndarray, 
                    y: np.ndarray, 
                    m: float, 
                    c: float) -> float:
        return ((y - m * x + c)**2).mean()

    # A part of gradient descent to calculate the current slope
    def __step_gradient_Descent(self, 
                                    x: np.ndarray, 
                                    y: np.ndarray, 
                                    m: float, 
                                    c: float) -> set:
        mslope, cslope, M = 0, 0, x.shape[0] 
        for i in range(M):
            mslope += ((2/M) * (y[i] - m * x[i] - c) * (-x[i]))  # Finding slope for m
            cslope += ((2/M) * (y[i] - m * x[i] - c) * -1)  # Finding slope for C
        
        return mslope, cslope   # Returning slopes
    
    # Function to call to process of gradient descent
    def __gradient_Descent(self, x: np.ndarray, y: np.ndarray):
        prev_Cost = 10000000    
        # Running through the iterations
        for count in range(self.__niterations):
            # Calculating the slope for current m and C
            mslope, cslope = self.__step_gradient_Descent(x, y, self.coef_, self.intercept_)

            # Finding new values of m and C
            new_coef = self.coef_ - self.__alpha * mslope
            new_intercept = self.intercept_ - self.__alpha * cslope

            # Finding cost
            current_Cost = self.__cost(x, y, new_coef, new_intercept)
            
            # If the current cost > previous cost, then we are
            # overshooting, we need to slow down the learning rate
            # hence, decreasing alpha by 1/10th value, else updating
            # m and C with current value
            if current_Cost > prev_Cost: self.__alpha /= 10
            else: self.coef_ = new_coef; self.intercept_ = new_intercept
            
            # If difference between the costs is too small
            # we can simply stop process
            if abs(prev_Cost - current_Cost) < self.__stoppage: break
            
            prev_Cost = current_Cost  # Updating previous cost with current cost
            
    # A fit function to train algorithm as per training set
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Calling gradient desecent process to train
        # algorithm, and set m and C values
        self.__gradient_Descent(x, y)    
        

    # Predict function to predict values for provided test data
    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.intercept_ and not self.coef_: return np.array([])
        y_pred = self.coef_ * x + self.intercept_
        return y_pred

    # Calculate the coefficient determination
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.coef_ and not self.intercept_: return float('-inf') 
        y_pred = self.predict(x)
        score = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
        return score

    # Calculate the avg cost or mean difference between
    # predicted values and actual values
    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.coef_ and not self.intercept_: return float('inf')
        y_pred = self.predict(x)
        return ((y - y_pred) ** 2).mean()