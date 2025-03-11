# orpML/deep_model.py
# Deep learning model for predicting Eh7 from microbial abundances
# 20250301 jmd

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

class DeepModel(L.LightningModule):
    def __init__(self, num_features):
        super().__init__()
        # Log train and val losses to different directories to plot together in TensorBoard
        logdir = "tb_logs"
        self.train_writer = SummaryWriter(os.path.join(logdir, "train"))
        self.val_writer = SummaryWriter(os.path.join(logdir, "val"))

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
        self.train_writer.add_scalar("mae", self.train_mae.compute(), self.current_epoch)
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        self.val_writer.add_scalar("mae", self.val_mae.compute(), self.current_epoch)
        self.val_mae.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_mae.update(out, y)

    def on_test_epoch_end(self):
        test_mae = self.test_mae.compute()
        self.log("test_mae", test_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.test_mae.reset()

class OrpMLDataModule(L.LightningDataModule): 
    def __init__(self): 
        super().__init__() 
        self.num_features = None
          
    def prepare_data(self): 
        pass

    def setup(self, stage=None): 
        # This reads the data into X_train, X_test, y_train, and y_test
        from .extract import X_train, X_test, y_train, y_test, metadata_train, metadata_test
        # This provides the preprocessing pipeline
        from .transform import preprocessor
        # Configure the pipeline to use abundances of phyla
        preprocessor.set_params(feat__abundance__use__rank="phylum", feat__Zc__use__rank="phylum")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Fit and transform the training data (the features)
            self.X_train = preprocessor.fit_transform(X_train)
            # Convert target values to NumPy array and reshape to have same number of dimensions as features (2D)
            self.y_train = y_train.to_numpy().reshape(-1, 1)
            # Create the dataset for PyTorch
            self.train_dataset = TensorDataset(torch.tensor(self.X_train).float(), torch.tensor(self.y_train).float())
            # Further split training data into train and val datasets
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [0.8, 0.2])

        # Assign test dataset for use in dataloader
        if stage == "test":
            self.X_test = preprocessor.transform(X_test)
            self.y_test = y_test.to_numpy().reshape(-1, 1)
            self.test_dataset = TensorDataset(torch.tensor(self.X_test).float(), torch.tensor(self.y_test).float())

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=100, num_workers=4, shuffle=True)
  
    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=100, num_workers=4)
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=100, num_workers=4)

# Instantiate the data module
dm = OrpMLDataModule()
# Get number of features from data
dm.prepare_data()
dm.setup(stage="fit")
num_features = dm.X_train.shape[1]
# Instantiate the deep learning model
model = DeepModel(num_features)
# Setup the trainer and fit the model to the data
trainer = L.Trainer(logger=False, enable_checkpointing=False, max_epochs=50)
trainer.fit(model, datamodule=dm)
# Test the model
dm.setup(stage="test")
trainer.test(model, datamodule=dm)
