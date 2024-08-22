# Linear Regression For 1D Array

A simple code for prediction of the values for 1D array. We can provide data in the form of csv files which contains the values as (X, Y), where X is the input and Y is the output for the particular row.

It uses simple mathematics functions to calculate or predict the values for X.

# Prerequisites
### Creating Virtual Enviroment:
```shell
conda create --name linear-regression-1d python=3.10
```
### Activating
```shell
conda activate linear-regression-1d
```
### Installing Packages
```shell
pip install -r requirements.txt
```
### Registrating the environment in a notebook
```shell
ipython kernel install --name "linear-regression-1d" --user
```

# Usage
Important: Only uses numeric features in 1D numpy array for the X and Y matrices.

Feel free to create a pull request with the additional implementation.

## Functions
- fit(x, y) :- This function trains the algorithm, calculates all the parameters required for predictions.
- predict(x) :- This function predicts the values for the given values in x, return a np.ndarray contains all the predicted values.
- score(x, y) :- This function calculates the coefficient of determination for the predicted values and actual values.

### Sample Code
```
from Linear_Regression_1D import Linear_Regression
training_data = np.text('training_data.csv', delimiter = ',')
test_data = np.text('test_data.csv', delimiter = ',')

X_Train = training_data[:, 0]
Y_Train = training_data[:, 1]
X_Test = test_data[:, 0]
Y_Test = test_data[:, 1]

algo = Linear_Regression()
algo.fit(X_Train, Y_Train)

Y_Pred = algo.predict(X_Test)

print(zip(Y_Test, Y_Pred))
print(algo.score(X_Test, Y_Test))
```