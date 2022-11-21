import numpy as np

# Create function to preprocess testing data make prediction and evaluate
def init(test_data, model, columns):
    # Preprocess test data 
    columns = list(columns)
    columns.remove("Class")
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]

    # Make prediction
    X_test = test_data.values
    y_pred = model.predict(X_test)

    # Calculate error
    difference = np.subtract(y_pred, X_test)
    squared_diff = np.square(difference)
    mse = squared_diff.mean()
    return mse