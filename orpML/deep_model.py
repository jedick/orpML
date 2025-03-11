# orpML/deep_model.py
# Deep learning model for predicting Eh7 from microbial abundances
# 20250301 jmd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader, random_split
from .extract import *
from .transform import *

import os
from torch.utils.tensorboard import SummaryWriter

# Log train and val losses to different directories to plot together in TensorBoard
LOG_DIR = "tb_logs"
train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
val_writer = SummaryWriter(os.path.join(LOG_DIR, "val"))

class DeepModel(L.LightningModule):
    def __init__(self, num_features):
        super().__init__()

        ## Simple linear model
        #self.layer = nn.Linear(num_features, 1)
        #self.learning_rate = 0.1

        self.layers = nn.Sequential(

            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
        )
        self.learning_rate = 0.005
        # Logging docs: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        # Torchmetrics in Lightning: https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        self.train_mae = torchmetrics.regression.MeanAbsoluteError()
        self.val_mae = torchmetrics.regression.MeanAbsoluteError()
        self.test_mae = torchmetrics.regression.MeanAbsoluteError()

        ## Initialize each layer's weights
        #nn.init.kaiming_uniform_(self.fc1.weight)
        #nn.init.kaiming_uniform_(self.fc2.weight)
        #nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_mae.update(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mae.update(out, y)
        return loss

    def on_train_epoch_end(self):
        train_writer.add_scalar("mae", self.train_mae.compute(), self.current_epoch)
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        val_writer.add_scalar("mae", self.val_mae.compute(), self.current_epoch)
        self.val_mae.reset()

# Setup the preprocessing pipeline to use abundances of phyla
preprocessor.set_params(feat__abundance__use__rank="phylum", feat__Zc__use__rank="phylum")
# Fit on training data and transform test data (the features)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
# Convert target values to NumPy array and reshape to have same number of dimensions as features (2D)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

# Create the dataset and dataloader for train and test folds
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
# Further split training data into train and val datasets
train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=100, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=100, num_workers=4, shuffle=False)
# Create test dataset
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Instantiate the model
num_features = X_train.shape[1]
model = DeepModel(num_features)

# Setup the trainer
trainer = L.Trainer(logger=False, enable_checkpointing=False, max_epochs=50)

# Train the model
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Model evaluation
# Set up MAE metric
mean_absolute_error = torchmetrics.MeanAbsoluteError()
# Put model in eval mode
model.eval()
# Iterate over test data batches with no gradients
with torch.no_grad():
  for x, y in test_dataloader:
    y_hat = model(x)
    # Update MAE metric
    mean_absolute_error(y_hat, y)
test_mae = mean_absolute_error.compute()
print(f"MeanAbsoluteError on test set: {test_mae}")

