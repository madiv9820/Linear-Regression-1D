'''
    Creating a Linear Regression Class from scratch. Linear Regression
    creates a linear relation by x and y.
    y = m * x + C
'''
import numpy as np

class Linear_Regression:
    # Constructor
    def __init__(self) -> None:
        self.intercept_ = None  # Intercept is the C
        self.coef_  = None      # Coef is the slope m

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Finding m and c, where the cost is the minimum
        # Cost = sum((y_actual - y_pred) ^ 2)
        # y_pred = m * x_actual + c
        # We have find these values using partial derivation

        self.coef_ = ((x*y).mean() - (x.mean() * y.mean())) / ((x**2).mean() - (x.mean())**2)
        self.intercept_ = y.mean() - self.coef_ * x.mean()

    def predict(self, x: np.ndarray) -> np.ndarray:
        # If any case, fit function has not been performed yet
        if not self.coef_ and not self.intercept_: return np.array([])

        # After finding m and c we can predict values
        y_pred = self.coef_ * x + self.intercept_
        return y_pred

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        # Finding the coefficient of determination
        # Coefficient of determination = 1 - (sum((y_actual - y_pred) ^ 2) / sum((y_actual - y_mean) ^ 2)))
        y_pred = self.predict(x)
        score = 1 - np.sum(((y_pred - y)**2))  / np.sum((y_pred - y.mean()) ** 2)
        return score