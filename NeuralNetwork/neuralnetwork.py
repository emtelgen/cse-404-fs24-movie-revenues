import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import joblib  # For saving the model
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # To save the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from letterbox import Letterboxd
from revenue import RevenueData

# Define a simple feedforward neural network in PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer (single value for regression)

    def forward(self, x):
        # Define the forward pass through the network
        x = torch.relu(self.fc1(x))  # ReLU activation after first layer
        x = torch.relu(self.fc2(x))  # ReLU activation after second layer
        x = self.fc3(x)  # Output layer
        return x

def train_and_evaluate_model(merged_data):
    """
    Trains a deep learning model (neural network) on the merged data.
    Evaluates the model performance and displays the results.
    """
    # Prepare the data
    X = merged_data.drop(columns=['revenue'])  # Drop unnecessary columns
    y = merged_data['revenue']

    bool_columns = X.select_dtypes(include=['bool']).columns
    X[bool_columns] = X[bool_columns].astype(float)
    print("Columns in X before conversion: ", X.columns)
    print("Data types in X before conversion: ", X.dtypes)

    # Convert data to numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape to column vector
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Reshape to column vector

    # Initialize the neural network model
    input_dim = X_train.shape[1]  # Number of features
    model = NeuralNetwork(input_dim=input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()

        y_train_pred = model(X_train_tensor)

        train_loss = criterion(y_train_pred, y_train_tensor)

        optimizer.zero_grad()
        train_loss.backward()

        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_tensor)
            test_loss = criterion(y_test_pred, y_test_tensor)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        test_pred = model(X_test_tensor)

    train_mae = mean_absolute_error(y_train, train_pred.numpy())
    test_mae = mean_absolute_error(y_test, test_pred.numpy())

    train_mse = mean_squared_error(y_train, train_pred.numpy())
    test_mse = mean_squared_error(y_test, test_pred.numpy())

    print(f"Training MAE: {train_mae}")
    print(f"Test MAE: {test_mae}")
    print(f"Training MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    torch.save(model.state_dict(), 'neural_network_model_best.pth')
    print("Best model saved to 'neural_network_model_best.pth'")

def main():
    """
    Main function to load data, merge it, and train the model.
    """
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    merged_data = pd.read_csv("letterboxdata.csv", nrows=1000000) 
    print(merged_data.columns)
    
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    merged_data = merged_data.dropna(subset=['revenue'])

    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)
    print(merged_data.dtypes)

    merged_data = merged_data.drop(columns=['name', 'tagline', 'name_crew', 'studio', 'description', 'role', 'type'])

    categorical_columns = merged_data.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_columns}")
    merged_data.to_csv('merged_data.csv', index=False)

    # One-Hot Encoding for categorical columns
    merged_data = pd.get_dummies(merged_data, columns=['genre', 'language', 'country'], drop_first=True)

    # Train and evaluate the model
    train_and_evaluate_model(merged_data)

if __name__ == "__main__":
    main()
