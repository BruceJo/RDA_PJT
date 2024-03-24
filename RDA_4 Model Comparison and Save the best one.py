from bson.binary import Binary
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling1D, Input, LayerNormalization, LSTM, MultiHeadAttention, ReLU, SimpleRNN
from keras.models import  Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

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
        'LSTM': load_model_from_name('LSTM'),
        'RNN': load_model_from_name('RNN'),
        'TCN': load_model_from_name('TCN', custom_objects={'TCN': TCN}),  # Custom objects for TCN layer
        'Transformer': load_model_from_name('Transformer')
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

def LSTM_model(look_back, input_features=2, optimizer_lr=0.005, dropout_rate=0.5, recurrent_dropout_rate=0.2,
               regularization_val=0.01):
    """
    Build and return an optimized LSTM model.

    Parameters:
    - look_back: Number of previous time steps to use as input variables.
    - input_features (default=2): Number of features in the input data.
    - optimizer_lr (default=0.005): Learning rate for the optimizer.
    - dropout_rate (default=0.5): Fraction of the input units to drop.
    - recurrent_dropout_rate (default=0.2): Fraction of the recurrent units to drop.
    - regularization_val (default=0.01): Regularization strength for L2 regularization.

    Returns:
    - model: Compiled LSTM model.
    """

    # Initialize a sequential model.
    model = Sequential()

    # Add the LSTM layer with dropout and recurrent dropout
    model.add(LSTM(64, input_shape=(look_back, input_features), return_sequences=True, dropout=dropout_rate,
                   recurrent_dropout=recurrent_dropout_rate))

    # Batch normalization layer
    model.add(BatchNormalization())

    # Dropout layer for regularization
    model.add(Dropout(dropout_rate))

    # Second LSTM layer
    model.add(LSTM(64, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))

    # Batch normalization layer
    model.add(BatchNormalization())

    # Dropout layer for regularization
    model.add(Dropout(dropout_rate))

    # Dense (fully connected) layer
    model.add(Dense(64, activation='relu'))

    # Output layer with L2 regularization
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(regularization_val)))

    # Compile the model with Adam optimizer and mean squared error loss function
    optimizer = Adam(learning_rate=optimizer_lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', run_eagerly=True)

    # Return the compiled model.
    return model

def RNN_model(look_back, input_features=2, optimizer_lr=0.005, dropout_rate=0.5, regularization_val=0.01):
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

def TCN_model(look_back, input_features=2, optimizer_lr=0.005, dropout_rate=0.5, regularization_val=0.01):
    """
    Build and return a TCN model.

    Parameters:
    - look_back: Number of previous time steps to use as input variables.
    - input_features (default=2): Number of features in the input data.
    - optimizer_lr (default=0.005): Learning rate for the optimizer.
    - dropout_rate (default=0.5): Fraction of the input units to drop.
    - regularization_val (default=0.01): Regularization strength for L2 regularization.

    Returns:
    - model: Compiled TCN model.
    """

    # Initialize a sequential model.
    model = Sequential()

    # Add the first TCN layer.
    # This layer has 64 filters, uses 4-sized kernels, and takes an input shape based on the look-back period and number of features.
    # 'return_sequences=True' indicates that the layer returns the full sequence to the next layer.
    model.add(TCN(nb_filters=64, kernel_size=4, input_shape=(look_back, input_features), return_sequences=True))

    # Add a second TCN layer with 64 filters and 4-sized kernels.
    # 'return_sequences=False' indicates that this layer returns condensed feature representations.
    model.add(TCN(nb_filters=64, kernel_size=4, return_sequences=False))

    # Batch normalization layer normalizes activations of the previous layer,
    # making the network train faster and achieve better accuracy.
    model.add(BatchNormalization())

    # Activation function layer. Here we're using ReLU (Rectified Linear Unit) activation.
    model.add(ReLU())

    # Dropout is a regularization method where randomly selected neurons are ignored during training.
    # This reduces overfitting by preventing complex co-adaptations on training data.
    model.add(Dropout(dropout_rate))

    # Add a dense (fully connected) layer with 64 neurons and ReLU activation.
    model.add(Dense(64, activation='relu'))

    # Output layer, which outputs a prediction with the same dimensionality as 1.
    # It uses linear activation (default for regression tasks) and L2 regularization to prevent overfitting.
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(regularization_val)))

    # Compile the model with an Adam optimizer (with a custom learning rate) and mean squared error loss function.
    # 'run_eagerly=True' makes debugging easier, but it might be slower. Consider setting it to 'False' for large datasets.
    optimizer = Adam(learning_rate=optimizer_lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', run_eagerly=True)

    # Return the compiled model.
    return model

def Transformer_model(look_back, input_features=2, num_heads=2, ff_dim=64, num_transformer_blocks=2, mlp_units=64, dropout_rate=0.1):
    """
    Build and return an optimized Transformer model.

    Parameters:
    - look_back: Number of previous time steps to use as input variables.
    - input_features (default=2): Number of features in the input data.
    - num_heads (default=2): Number of attention heads in the multi-head attention layer.
    - ff_dim (default=64): Dimensionality of the feed-forward layer.
    - num_transformer_blocks (default=2): Number of transformer blocks in the model.
    - mlp_units (default=64): Number of units in the MLP head for final prediction.
    - dropout_rate (default=0.1): Fraction of the input units to drop.

    Returns:
    - model: Compiled and optimized Transformer model.
    """

    inputs = Input(shape=(look_back, input_features))
    x = inputs

    # Stacked Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-Head Self-Attention layer
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_features, dropout=dropout_rate)(x, x)
        attention_output = LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed-Forward layer
        ff_output = Dense(ff_dim, activation='relu')(attention_output)
        ff_output = Dense(input_features)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    # Final layers
    output = GlobalAveragePooling1D()(x)
    output = Dense(mlp_units, activation='relu')(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(1, activation='linear')(output)

    model = Model(inputs=inputs, outputs=output)

    # Learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

def train_LSTM_model(X_train, Y_train, X_val, Y_val, look_back=10):
    # Create the LSTM model using the defined function.
    model = LSTM_model(look_back)

    # Define training callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32,
                        callbacks=[early_stop, reduce_lr])

    Save_model(model, "LSTM")

    return model, history

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

def train_TCN_model(X_train, Y_train, X_val, Y_val, look_back=10):
    # Create the TCN model using the defined function.
    model = TCN_model(look_back)

    # Define training callbacks.
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model.
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32,
                        callbacks=[early_stop, reduce_lr])

    Save_model(model, "TCN")

    return model, history

def train_Transformer_model(X_train, Y_train, X_val, Y_val, look_back=10):
    # Create the Transformer model using the defined function.
    model = Transformer_model(look_back)

    # Define training callbacks.
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model.
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32,
                        callbacks=[early_stop, reduce_lr])

    Save_model(model, "Transformer")

    return model, history

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
look_back = 10

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

# Train your models and then save them:
LSTM_model, LSTM_history = train_LSTM_model(X_train, Y_train, X_val, Y_val, look_back=10)
RNN_model, RNN_history = train_RNN_model(X_train, Y_train, X_val, Y_val, look_back=10)
TCN_model, TCN_history = train_TCN_model(X_train, Y_train, X_val, Y_val, look_back=10)
Transformer_model, Transformer_history = train_Transformer_model(X_train, Y_train, X_val, Y_val, look_back=10)

# Get the sorted list of models based on MSE
sorted_models = select_best_model(X_test, Y_test)

# Unpack information for each model
best_model_name, best_model, best_model_error, best_model_predictions = sorted_models[0]
print(f"The best model is: {best_model_name} with MSE: {best_model_error}")

second_best_model_name, second_best_model, second_best_model_error, second_best_model_predictions = sorted_models[1]
print(f"The second best model is: {second_best_model_name} with MSE: {second_best_model_error}")

third_best_model_name, third_best_model, third_best_model_error, third_best_model_predictions = sorted_models[2]
print(f"The third best model is: {third_best_model_name} with MSE: {third_best_model_error}")

worst_model_name, worst_model, worst_model_error, worst_model_predictions = sorted_models[-1]
print(f"The worst model is: {worst_model_name} with MSE: {worst_model_error}")

# Create a temporary array with the same number of columns as the original data
temp_predicted = np.zeros((best_model_predictions.shape[0], data_normalized.shape[1]))
temp_actual = np.zeros((Y_test.shape[0], data_normalized.shape[1]))

# Fill the first column (or whichever column corresponds to the prediction) with predicted values
temp_predicted[:, 0] = best_model_predictions.ravel()
temp_actual[:, 0] = Y_test.ravel()

# Now inverse transform
original_predictions = scaler.inverse_transform(temp_predicted)[:, 0]
original_actual = scaler.inverse_transform(temp_actual)[:, 0]

# Compute the Mean Squared Error (MSE) between the de-normalized actual and predicted values.
mse_denormalized = mean_squared_error(original_actual, original_predictions)
print(f"Mean Squared Error on de-normalized data: {mse_denormalized}")

# Plot a comparison between predicted and actual values
plt.figure(figsize=(15, 6))
plt.plot(original_actual, label="Actual values", color='blue')
plt.plot(original_predictions, label="Predicted values of " + sorted_models[0][0], color='red', alpha=0.7)
plt.title("Prediction vs Actual values")
plt.legend()
plt.savefig("comparison_plot.png", dpi=300)
plt.show()

best_model_type = best_model_name
db.save_model(best_model, best_model_name, best_model_type)