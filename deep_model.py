# orpML/nn_model.py
# Deep learning model for predicting Eh7 from microbial abundances
# 20250301 jmd

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import MeanAbsoluteError
from extract import *
from transform import *

# Setup the preprocessing pipeline to get abundances of phyla
preprocessor.set_params(selectfeatures__abundance = "phylum", keeptoptaxa__n = 150)
# Note: the imputer step of the transform yields a NumPy array with 2 dimensions
X_train = preprocessor.fit_transform(X_train)
# Transform test data after fitting on the training data
X_test = preprocessor.transform(X_test)
# Convert Eh7 values to NumPy array and reshape it to 2 dimensions
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

# Create the dataset and dataloader for train and test folds
dataset_train = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
dataloader_train = DataLoader(dataset_train, batch_size = 100, shuffle = True)
dataset_test = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
dataloader_test = DataLoader(dataset_test, batch_size = 100)

class DeepModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        ## Initialize each layer's weights
        #nn.init.kaiming_uniform_(self.fc1.weight)
        #nn.init.kaiming_uniform_(self.fc2.weight)
        #nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        return self.layers(x)

class LinearModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.layer(x)

# Instantiate the desired model
num_features = X_train.shape[1]
model = DeepModel(num_features)

# Use higher learning rate for linear model
if model.__class__.__name__ == "DeepModel":
    lr = 0.01
if model.__class__.__name__ == "LinearModel":
    lr = 0.1

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

# The training loop
# Make sure the model is in training mode
model.train()
num_epochs = 2000
for epoch in range(num_epochs):
    running_loss = 0.0
    running_size = 0.0
    for data in dataloader_train:
        # Set the gradients to zero
        optimizer.zero_grad()
        # Get features and target from the dataloader
        features, target = data
        # Run a forward pass
        predictions = model(features)
        # Compute loss
        loss = criterion(predictions, target)
        # Compute new gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Keep track of the loss weighted by number of samples in each batch
        running_loss += loss.item() * target.shape[0]
        running_size += target.shape[0]

    # Calculate overall loss for epoch
    epoch_loss = running_loss / running_size
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        print(f"Epoch: {epoch:03d}, Loss: {epoch_loss:.2f}")

# Model evaluation
# Set up MAE metric
mean_absolute_error = MeanAbsoluteError()
# Put model in eval mode
model.eval()
# Iterate over test data batches with no gradients
with torch.no_grad():
  for features, target in dataloader_test:
    predictions = model(features)
    # Update MAE metric
    mean_absolute_error(predictions, target)
mae = mean_absolute_error.compute()
print(f"MeanAbsoluteError: {mae}")

