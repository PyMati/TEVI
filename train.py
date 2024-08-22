from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # x shape: (b*t, c, h, w)
        x = self.normalize(x)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


def temporal_loss(x):
    # x shape: (b, t, c, h, w)
    return F.mse_loss(x[:, 1:], x[:, :-1])


class NoiseScheduler:
    def __init__(self, num_timesteps, initial_noise=1e-4, final_noise=0.02):
        self.num_timesteps = num_timesteps
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.betas = self._linear_schedule()

    def _linear_schedule(self):
        # Linearly interpolate between the initial and final noise values
        return torch.linspace(self.initial_noise, self.final_noise, self.num_timesteps)

    def get_variance(self, t):
        # Get the variance corresponding to the timestep t
        # t is expected to be a tensor of shape (batch_size,)
        return self.betas[(t * (self.num_timesteps - 1)).long()]


def diffusion_loss(model, x_0, t, noise, perceptual_loss_fn, noise_scheduler, lambda_perceptual=0.1, lambda_temporal=0.1):
    b, t_, c, h, w = x_0.shape
    
    # Get the variance from the noise scheduler
    variance = noise_scheduler.get_variance(t)
    
    # Add noise to the input
    x_noisy = x_0 + noise * torch.sqrt(variance.view(b, 1, 1, 1, 1))
    
    # Predict the noise
    predicted_noise = model(x_noisy)
    
    # Calculate MSE loss
    mse_loss = F.mse_loss(noise, predicted_noise)
    
    # Calculate perceptual loss
    x_0_flat = x_0.view(b*t_, c, h, w)
    x_denoised_flat = (x_noisy - predicted_noise).view(b*t_, c, h, w)
    
    vgg_x0 = perceptual_loss_fn(x_0_flat)
    vgg_denoised = perceptual_loss_fn(x_denoised_flat)
    
    perceptual_loss = sum([F.mse_loss(vx0, vd) for vx0, vd in zip(vgg_x0, vgg_denoised)])
    
    # Calculate temporal loss
    temp_loss = temporal_loss(x_0)
    
    # Combine losses
    total_loss = mse_loss + lambda_perceptual * perceptual_loss + lambda_temporal * temp_loss
    
    return total_loss, mse_loss, perceptual_loss, temp_loss


def train_diffusion_model(model, train_dataset, val_dataset, num_epochs, batch_size, learning_rate, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    noise_scheduler = NoiseScheduler(num_timesteps=25)  # example with 1000 timesteps
    perceptual_loss_fn = PerceptualLoss().to(device)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_perceptual_loss = 0.0
        train_temporal_loss = 0.0
        
        for videos, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            videos = videos.to(device)
            batch_size = videos.shape[0]
            
            # Sample random timesteps
            t = torch.rand(batch_size, device=device)
            
            # Generate random noise
            noise = torch.randn_like(videos)
            
            optimizer.zero_grad()
            loss, mse_loss, perceptual_loss, temp_loss = diffusion_loss(
                model, videos, t, noise, perceptual_loss_fn, noise_scheduler
            )
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_perceptual_loss += perceptual_loss.item()
            train_temporal_loss += temp_loss.item()
        
        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_perceptual_loss /= len(train_loader)
        train_temporal_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        val_perceptual_loss = 0.0
        val_temporal_loss = 0.0
        with torch.no_grad():
            for videos, _ in val_loader:
                videos = videos.to(device)
                batch_size = videos.shape[0]
                t = torch.rand(batch_size, device=device)
                noise = torch.randn_like(videos)
                loss, mse_loss, perceptual_loss, temp_loss = diffusion_loss(
                    model, videos, t, noise, perceptual_loss_fn, noise_scheduler
                )
                val_loss += loss.item()
                val_mse_loss += mse_loss.item()
                val_perceptual_loss += perceptual_loss.item()
                val_temporal_loss += temp_loss.item()
        
        val_loss /= len(val_loader)
        val_mse_loss /= len(val_loader)
        val_perceptual_loss /= len(val_loader)
        val_temporal_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Total: {train_loss:.4f}, MSE: {train_mse_loss:.4f}, Perceptual: {train_perceptual_loss:.4f}, Temporal: {train_temporal_loss:.4f}")
        print(f"Val - Total: {val_loss:.4f}, MSE: {val_mse_loss:.4f}, Perceptual: {val_perceptual_loss:.4f}, Temporal: {val_temporal_loss:.4f}")
        
        scheduler.step()
    
    return model
