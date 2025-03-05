# orpML/deep_model.py
# Deep learning model for predicting Eh7 from microbial abundances
# 20250301 jmd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
from extract import *
from transform import *

# Setup the preprocessing pipeline to get abundances of phyla
preprocessor.set_params(selectfeatures__abundance = "phylum", keeptoptaxa__n = 150)
#preprocessor.set_params(selectfeatures__abundance = "phylum", selectfeatures__Zc = "domain", keeptoptaxa__n = 150)
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

class DeepModel(L.LightningModule):
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
        self.learning_rate = 0.01

        ## Initialize each layer's weights
        #nn.init.kaiming_uniform_(self.fc1.weight)
        #nn.init.kaiming_uniform_(self.fc2.weight)
        #nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = F.mse_loss(output, target)
        # https://stackoverflow.com/questions/71236391/pytorch-lightning-print-accuracy-and-loss-at-the-end-of-each-epoch
        self.log("train_loss", loss, prog_bar = True, on_step = False, on_epoch = True)
        return loss

class LinearModel(L.LightningModule):
    def __init__(self, num_features):
        super().__init__()
        self.layer = nn.Linear(num_features, 1)
        self.learning_rate = 0.1

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = F.mse_loss(output, target)
        self.log("train_loss", loss, prog_bar = True, on_step = False, on_epoch = True)
        return loss

# Instantiate the desired model
num_features = X_train.shape[1]
model = DeepModel(num_features)
# Setup the trainer
# Adjust refresh rate for progress bar:
# https://lightning.ai/docs/pytorch/stable/common/progress_bar.html
trainer = L.Trainer(max_epochs = 1000, callbacks = [TQDMProgressBar(refresh_rate = 100)])

# Train the model
trainer.fit(model, train_dataloaders = dataloader_train)

# Model evaluation
# Set up MAE metric
mean_absolute_error = torchmetrics.MeanAbsoluteError()
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

