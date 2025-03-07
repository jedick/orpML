# orpML/deep_model.py
# Deep learning model for predicting Eh7 from microbial abundances
# 20250301 jmd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from .extract import *
from .transform import *

# Setup the preprocessing pipeline to use abundances of phyla
preprocessor.set_params(feat__abundance__use__rank = "phylum", feat__Zc__use__rank = "phylum")
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
train_dataloader = DataLoader(train_dataset, batch_size = 20, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 20, shuffle = False)
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
test_dataloader = DataLoader(test_dataset, batch_size = 20, shuffle = False)

class DeepModel(L.LightningModule):
    def __init__(self, num_features):
        super().__init__()

        ## Simple linear model
        #self.layer = nn.Linear(num_features, 1)
        #self.learning_rate = 0.1

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
        # Logging docs: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        # Torchmetrics in Lightning: https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        self.train_mae = torchmetrics.regression.MeanAbsoluteError()
        self.val_mae = torchmetrics.regression.MeanAbsoluteError()
        self.test_mae = torchmetrics.regression.MeanAbsoluteError()
        ## NOTUSED: combining outputs in each epoch
        ## https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation

        ## Initialize each layer's weights
        #nn.init.kaiming_uniform_(self.fc1.weight)
        #nn.init.kaiming_uniform_(self.fc2.weight)
        #nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        # https://stackoverflow.com/questions/71236391/pytorch-lightning-print-accuracy-and-loss-at-the-end-of-each-epoch
        self.log("train_loss", loss, prog_bar = True, on_step = False, on_epoch = True)
        self.train_mae.update(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("val_loss", loss, prog_bar = True, on_step = False, on_epoch = True)
        self.val_mae.update(out, y)
        return loss

    def on_train_epoch_end(self):
        self.log("train_mae", self.train_mae.compute())
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae.compute())
        self.val_mae.reset()

# Instantiate the model
num_features = X_train.shape[1]
model = DeepModel(num_features)

# Setup the trainer
# Use tensorboard logger: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html
logger = TensorBoardLogger("tb_logs", name = "orpML")
# Adjust refresh rate for progress bar: https://lightning.ai/docs/pytorch/stable/common/progress_bar.html
#trainer = L.Trainer(max_epochs = 100, callbacks = [TQDMProgressBar(refresh_rate = 100)], logger = logger)
trainer = L.Trainer(max_epochs = 100, logger = logger)

# Train the model
trainer.fit(model, train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)

# Model evaluation
# Set up MAE metric
mean_absolute_error = torchmetrics.MeanAbsoluteError()
# Put model in eval mode
model.eval()
# Iterate over test data batches with no gradients
with torch.no_grad():
  for features, target in test_dataloader:
    predictions = model(features)
    # Update MAE metric
    mean_absolute_error(predictions, target)
mae = mean_absolute_error.compute()
print(f"MeanAbsoluteError: {mae}")

