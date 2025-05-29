import lightning as pl
from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch.nn as nn
import torch

# Deal with input images of 600 x 600
class HighwayEnvModel(pl.LightningModule):

    def __init__(self, n_channels, height, width, latent_dim, loss_fn=F.mse_loss, lr=1e-3, eps=1e-8, weight_decay=0, seed=42):
        super(HighwayEnvModel, self).__init__()
        
        self.save_hyperparameters()

        self.total_train_loss = 0
        self.num_train_batches = 0

        self.total_val_loss = 0
        self.num_val_batches = 0

        self.loss_fn = loss_fn

        self.conv_kernel = 5
        self.stride = 3
        self.padding = 1

        print(f"Height: {height}")
        print(f"Width: {width}")
        # First Conv2D
        self.height_first_conv = ((height - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        self.width_first_conv =  ((width - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        print(f"After first conv: {self.height_first_conv} - {self.width_first_conv}")
        # Second Conv2D
        self.height_second_conv = ((self.height_first_conv - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        self.width_second_conv =  ((self.width_first_conv - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        print(f"After second conv: {self.height_second_conv} - {self.width_second_conv}")
        # Third Conv2D
        self.height_third_conv = ((self.height_second_conv - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        self.width_third_conv =  ((self.width_second_conv - self.conv_kernel + (2*self.padding)) // self.stride) + 1
        print(f"After third conv: {self.height_third_conv} - {self.width_third_conv}")
        # Final dim
        self.final_dim = latent_dim * self.height_third_conv * self.width_third_conv
        print(f"Final dim: {self.final_dim}")

        self.encoder = self.init_encoder(n_channels, latent_dim)
        self.decoder = self.init_decoder(n_channels, latent_dim)

    def init_encoder(self, n_channels, latent_dim):
        encoder = nn.Sequential(
            # First conv
            nn.Conv2d(n_channels, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # Second conv
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # Third conv
            nn.Conv2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
            # Linear
            nn.Linear(self.final_dim, latent_dim),
            nn.ReLU()
        )

        return encoder
        

    def init_decoder(self, n_channels, latent_dim):
        decoder = nn.Sequential(
            # First linear layer
            nn.Linear(latent_dim, self.final_dim),
            nn.ReLU(),
            # Reshape
            nn.Unflatten(1, (latent_dim, self.height_third_conv, self.width_third_conv)),
            # First conv transpose
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # Second conv transpose
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=self.stride-1),
            nn.ReLU(),
            # Third conv transpose
            nn.ConvTranspose2d(latent_dim // 2, n_channels, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.Sigmoid()
        )

        return decoder
    
    def forward(self, x, return_encodings=False):
        x = self.encoder(x)
        if not return_encodings:
            x = self.decoder(x)
            
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.eps, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.total_train_loss += loss
        self.num_train_batches += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.total_val_loss += loss
        self.num_val_batches += 1

    def test_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)

    def on_train_epoch_end(self):

        avg_train_loss = self.total_train_loss / self.num_train_batches

        self.log("train_loss", avg_train_loss.item())
        print(f"Epoch [{self.current_epoch}/{self.trainer.max_epochs}], Train loss: {avg_train_loss}")

    def on_validation_epoch_end(self):

        avg_val_loss = self.total_val_loss / self.num_val_batches

        self.log("val_loss", avg_val_loss.item())
        print(f"Epoch [{self.current_epoch}/{self.trainer.max_epochs}], Val loss: {avg_val_loss}")