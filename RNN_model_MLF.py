from bson.binary import Binary
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling1D, Input, LayerNormalization, MultiHeadAttention, ReLU, SimpleRNN
from keras.models import  Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def create_dataset(X, y, look_back=1):
    """
    Create dataset for time-series forecasting.

    Parameters:
    - X: Input time-series data (features).
    - y: Output time-series data (target).
    - look_back (default=1): Number of previous time steps to use as input variables
                             to predict the next time step.

    Returns:
    - dataX: List of the input sequences.
    - dataY: List of the output sequences.
    """

    dataX, dataY = [], []  # Initialize empty lists to hold our transformed sequences.

    # For each possible sequence in the input data...
    for i in range(len(X) - look_back):
        # Extract a sequence of 'look_back' features from the input data.
        sequence = X[i:(i + look_back), :]
        dataX.append(sequence)

        # Extract the output for this sequence from the 'y' data.
        output = y[i + look_back]
        dataY.append(output)

    # Convert the lists into NumPy arrays for compatibility with most ML frameworks.
    return np.array(dataX), np.array(dataY)

def Save_model(model, model_name, root_folder="saved_models"):
    """
    Save a given model's architecture as a JSON file and weights as an H5 file.

    Parameters:
    - model: Trained model to save.
    - model_name: Name of the model (e.g., "LSTM", "RNN").
    - root_folder (default='saved_models'): Name of the root folder where model subfolders will be created.

    Returns:
    - None
    """
    # Define the model-specific directory path
    model_dir = os.path.join(root_folder, model_name)

    # Ensure the save directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model architecture as a JSON file
    model_json_path = os.path.join(model_dir, f"{model_name}.json")
    with open(model_json_path, "w") as json_file:
        json_file.write(model.to_json())

    # Save the model weights as an H5 file
    model_weights_path = os.path.join(model_dir, f"{model_name}.h5")
    model.save_weights(model_weights_path)

    print(f"Saved {model_name} model to {model_dir}.")

# Internal function to load a model given its name
def load_model_from_name(model_name, custom_objects=None):
    """
    Load a model from its architecture JSON and weights file.

    Parameters:
    - model_name: The name of the model directory.
    - custom_objects: Any custom objects used in the model, like custom layers.

    Returns:
    - model: The loaded Keras model.
    """
    model_dir = os.path.join('saved_models', model_name)
    model_json_path = os.path.join(model_dir, f"{model_name}.json")
    model_weights_path = os.path.join(model_dir, f"{model_name}.h5")

    # Load model architecture from JSON file
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects=custom_objects)

    # Load model weights
    model.load_weights(model_weights_path)

    return model

def select_best_model(X_test, Y_test):
    """
    Load four different models and make predictions on the test data.
    Returns a sorted list of models based on Mean Squared Error (MSE).

    Parameters:
    - X_test: Test data used for predictions.
    - Y_test: True target values.

    Returns:
    - sorted_models: A list of tuples, each containing the model name, model instance, MSE, and predicted values.
    """

    # Dictionary defining the models to load, including any custom objects required
    models = {
        'RNN': load_model_from_name('RNN')
    }

    results = []

    # Iterate through each model, make predictions, and compute MSE
    for model_name, model in models.items():
        predicted_values = model.predict(X_test)
        error = mean_squared_error(Y_test, predicted_values)  # Result is numpy.float64
        results.append((model_name, model, error, predicted_values))

    # Sort the results based on MSE
    sorted_models = sorted(results, key=lambda x: x[2])

    return sorted_models

def RNN_model(look_back, input_features=2, optimizer_lr=0.001, dropout_rate=0.1, regularization_val=0.01):
    """
    Build and return a RNN model for time-series prediction.

    Parameters:
    - look_back: Number of previous time steps to use as input variables.
    - input_features (default=2): Number of features in the input data.
    - optimizer_lr (default=0.005): Learning rate for the optimizer.
    - dropout_rate (default=0.5): Fraction of the input units to drop.
    - regularization_val (default=0.01): Regularization strength for L2 regularization.

    Returns:
    - model: Compiled GRU model.
    """

    # Initialize a sequential model.
    model = Sequential()

    # Add the GRU layer with 64 units. The input shape is based on the look-back period and number of features.
    model.add(SimpleRNN(64, input_shape=(look_back, input_features), return_sequences=True, dropout=dropout_rate,
                        recurrent_dropout=0.2))

    # Add a second GRU layer with 64 units.
    model.add(SimpleRNN(64, return_sequences=False, dropout=dropout_rate, recurrent_dropout=0.2))

    # Batch normalization layer normalizes activations of the previous layer.
    model.add(BatchNormalization())

    # Activation function layer using ReLU activation.
    model.add(ReLU())

    # Dense layer for further processing.
    model.add(Dense(64, activation='relu'))

    # Output layer, which outputs a prediction. It uses linear activation (default for regression tasks) and L2 regularization.
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(regularization_val)))

    # Compile the model with an Adam optimizer (with a custom learning rate) and mean squared error loss function.
    optimizer = Adam(learning_rate=optimizer_lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', run_eagerly=True)

    # Return the compiled model.
    return model

def train_RNN_model(X_train, Y_train, X_val, Y_val, look_back=10):
    # Create the RNN model using the defined function.
    model = RNN_model(look_back)

    # Define training callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32,
                        callbacks=[early_stop, reduce_lr])

    Save_model(model, "RNN")

    return model, history


def evaluate_rnn(model, X_test, y_test):
    # Evaluate the model
    mse = model.evaluate(X_test, y_test)
    print("Evaluation Loss:", mse)
    
    mlflow.log_metric("mse", mse)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate MAE
    eval_mae = np.mean(np.abs(y_pred - y_test))
    print("Evaluation MAE:", eval_mae)
    
    # Log evaluation MAE to mlflow
    mlflow.log_metric("eval_mae", eval_mae)
    
    # Plot Predictions vs True values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.show()
    
    # Log scatter plot to mlflow
    plt.savefig("scatter_plot.png")
    mlflow.log_artifact("scatter_plot.png")
    plt.close()


class MongoDatabase:
    # Initializer method, called when a new instance of MongoDatabase is created
    def __init__(self):
        # Connection string for MongoDB
        CONNECTION_STRING = "mongodb://netdb:netdb3230!@10.255.93.173:27017/"
        # Creating MongoClient object using the connection string
        self.client = MongoClient(CONNECTION_STRING)

    def _fetch_data(self, collection_name, limit=None):
        """Private method to fetch data from a specified collection in MongoDB."""
        try:
            collection = self.client["TestAPI"][collection_name]
            cursor = collection.find({}).limit(limit) if limit else collection.find({})
            return pd.DataFrame(list(cursor))
        except Exception as e:
            print(f"Error while fetching data from {collection_name}: {e}")
            return None

    def get_environment(self, limit=None):
        """Public method to fetch environment data from the 'GH2' collection."""
        return self._fetch_data("GH2", limit)

    def get_growth(self, limit=None):
        """Public method to fetch growth data from the 'hydroponics_length1' collection."""
        return self._fetch_data("hydroponics_length1", limit)

    def save_model(self, model, model_name, model_type):
        """Method to save a model to MongoDB. It saves the model's HDF5 file."""
        model_file = f"{model_name}.h5"
        model.save(model_file)

        # Read and store the HDF5 file data
        with open(model_file, 'rb') as file:
            model_data = file.read()

        db = self.client["Things_to_refer"]
        collection = db["Previous_model_features"]

        # Create a document with model information
        model_document = {
            "name": model_name,
            "type": model_type,
            "model_data": Binary(model_data)
        }

        # Check if a model with the same name exists and update it, else insert a new document
        existing_document = collection.find_one({"name": model_name})
        if existing_document:
            collection.update_one({"_id": existing_document["_id"]}, {"$set": model_document})
            print(f"Existing model '{model_name}' updated in MongoDB.")
        else:
            collection.insert_one(model_document)
            print(f"New model '{model_name}' inserted into MongoDB.")


# Create an instance of the MongoDatabase class
db = MongoDatabase()

# Fetch growth data using the 'get_growth' method from the 'db' object
growth_data_1 = db.get_growth()
growth_data_2 = growth_data_1.drop(columns=['_id', 'date', 'sample_num',
                                   'plant_height              (㎝)', 'plant_diameter           (㎜)', 'leaflet          (cm)',
                                   'leaf_width         (cm)', 'last_flower_point         (th)',
                                   'growing_point_to_flower_point        (㎝)', 'note'], errors='ignore')

# Fetch environment data using the 'get_environment' method from the 'db' object.
environment_data_1 = db.get_environment(limit = 31200)

# Modify the 'environment_data_1' DataFrame to drop specified columns.
# environment_data_2 = environment_data_1.drop(columns=['_id', 'id', 'inFacilityId', 'sensorNo', 'sensingAt'], errors='ignore')
environment_data_2 = environment_data_1.drop(columns=['_id', 'id', 'inFacilityId', 'sensorNo', 'sensingAt', 'co2'], errors='ignore')
environment_averaged = environment_data_2.groupby(environment_data_2.index // 100).mean(numeric_only=True).reset_index(drop=True)

# Merge the 'environment_averaged' DataFrame and 'growth_data_2' DataFrame based on their indices.
training_data = pd.merge(environment_averaged, growth_data_2, left_index=True, right_index=True)

# Initialize the MinMaxScaler.
scaler = MinMaxScaler()
# 'data_normalized' will be a NumPy array where each feature (column) of the input data is normalized to the range [0, 1].
data_normalized = scaler.fit_transform(training_data)

# Assuming the last column of 'data_normalized' is the target variable that want to predict.
# 'data_normalized' is a 2D array with rows as individual data records and columns as features.

# Extract input features (every column except the last one).
X_data = data_normalized[:, :-1]

# Extract target variable (just the last column).
y_data = data_normalized[:, -1]

# Define the look-back period, which determines the number of past observations
# each input sequence will contain when transforming the data.
look_back = 24

# Transform the data into sequences of input (X) and output (Y) using the 'create_dataset' function.
X, Y = create_dataset(X_data, y_data, look_back)

# Define the size of the training set as 80% of the total data.
train_size = int(len(X) * 0.8)

# Split the data based on order (important for time series data).
# The first 80% is used for training.
X_train, X_temp = X[:train_size], X[train_size:]
Y_train, Y_temp = Y[:train_size], Y[train_size:]

# The remaining 20% is further divided into validation and test sets, each taking 10%.
# Split the remaining data into half for validation and testing.
val_size = len(X_temp) // 2

# Extract validation and test sets from the remaining data.
X_val, X_test = X_temp[:val_size], X_temp[val_size:]
Y_val, Y_test = Y_temp[:val_size], Y_temp[val_size:]

# Use MLFLOW
mlflow.set_experiment("rnn_experiment")

with mlflow.start_run():

    # Define the RNN model
    RNN_model = Sequential()
    RNN_model.add(SimpleRNN(64, input_shape=(look_back, 2), return_sequences=True))
    RNN_model.add(SimpleRNN(64))
    RNN_model.add(Dense(1))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    RNN_model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = RNN_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100)

    # Evaluate the model
    mse = RNN_model.evaluate(X_train, Y_train, verbose=0)
    mlflow.log_metric("mse", mse)

    # Save the model
    print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.sklearn.log_model(RNN_model, "rnn_model")    

    # Make predictions on the test set
    predictions = RNN_model.predict(X_test)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(Y_test, predictions)
    print("Mean Squared Error (MSE):", mse)

    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - Y_test))
    mlflow.log_metric("mae", mae)
    print("Mean Absolute Error (MAE):", mae)

    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    mlflow.log_metric("rmse", rmse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Visualize predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, predictions, color='blue', alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig("true_vs_predicted_values.png")
    plt.close()

    # Log scatter plot to mlflow
    mlflow.log_artifact("true_vs_predicted_values.png")

    # Plot a comparison between predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, label="Actual values", color='blue', alpha=0.5)
    plt.plot(predictions, label="Predicted values of RNN", color='red', alpha=0.5)
    plt.title("Prediction vs Actual values")
    plt.savefig("comparison_plot.png")
    plt.close() 

    # Log comparison plot to mlflow
    mlflow.log_artifact("comparison_plot.png")

    # Make predictions on the train set
    train_predictions = RNN_model.predict(X_train)

    # Visualize predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_train, train_predictions, color='blue', alpha=0.5)
    plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig("true_vs_predicted_values_on_train_data.png")
    plt.close()

    # Log scatter plot to mlflow
    mlflow.log_artifact("true_vs_predicted_values_on_train_data.png")

    # Plot a comparison between predicted and actual values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_train, label="Actual values", color='blue', alpha=0.5)
    plt.plot(train_predictions, label="Predicted values of RNN", color='red', alpha=0.5)
    plt.title("Prediction vs Actual values")
    plt.savefig("comparison_plot_on_train_data.png")
    plt.close() 

    # Log comparison plot to mlflow
    mlflow.log_artifact("comparison_plot_on_train_data.png")

mlflow.end_run()