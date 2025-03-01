# orpML/nn_model.py
# Deep learning model for predicting Eh7 from microbial abundances
# 20250301 jmd

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanAbsoluteError
from extract import *
from transform import *

# Setup the preprocessing pipeline to get abundances of phyla
preprocessor.set_params(selectfeatures__abundance = "phylum")
# Note: the imputer step of the transform yields a NumPy array
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.fit_transform(X_test)
# Convert Eh7 values to NumPy array
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Create the dataset and dataloader for train and test folds
dataset_train = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
dataloader_train = DataLoader(dataset_train, batch_size = 100, shuffle = True)
dataset_test = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
dataloader_test = DataLoader(dataset_test, batch_size = 100, shuffle = True)

# Build the model
#model = nn.Sequential(
#    nn.Linear(num_features, 20),
#    nn.ReLU(),
#    nn.Linear(20, 1),
#)

class Model(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.layer1 = nn.Linear(num_features, 20)
    self.act1 = nn.ReLU()
    self.layer2 = nn.Linear(20, 1)
    # Initialize each layer's weights
    nn.init.kaiming_uniform_(self.layer1.weight)
    nn.init.kaiming_uniform_(self.layer2.weight)

  def forward(self, x):
    x = self.act1(self.layer1(x))
    x = self.layer2(x)
    return(x)

# Instantiate the model with the number of features
num_features = X_train.shape[1]
model = Model(num_features)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# The training loop
# Make sure the model is in training mode
model.train()
num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader_train:
        # Set the gradients to zero
        optimizer.zero_grad()
        # Get features and target from the dataloader
        features, target = data
        # Unsqueeze the target to add batch dimension
        target = target.unsqueeze(1)
        # Run a forward pass
        predictions = model(features)
        # Compute loss
        loss = criterion(predictions, target)
        # Compute new gradients
        loss.backward()
        # Update the parameters
        optimizer.step()

    # Print information out
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


# Model evaluation
# Set up MAE metric
mean_absolute_error = MeanAbsoluteError()
# Put model in eval mode
model.eval()
# Iterate over test data batches with no gradients
with torch.no_grad():
  for features, target in dataloader_test:
    target = target.unsqueeze(1)
    predictions = model(features)
    # Update MAE metric
    mean_absolute_error(predictions, target)
mae = mean_absolute_error.compute()
print(f"MeanAbsoluteError: {mae}")

# Implementing validation:
# https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/
