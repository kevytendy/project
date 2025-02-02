import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import os
import sys

# Add the synthesizer directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_selection import evaluate_model, save_best_model
from evaluation import compute_statistical_similarity, compute_downstream_performance

# Compute KL loss (not used now)
def compute_kl(real, pred):
    return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, col_type, col_dim):
    kl = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        dim = col_dim[i]
        sta = end
        end = sta + dim
        fakex = x_fake[:, sta:end]
        realx = x_real[:, sta:end]
        if col_type[i] == "gmm":
            fake2 = fakex[:, 1:]
            real2 = realx[:, 1:]
            dist = torch.sum(fake2, dim=0) / torch.sum(fake2)
            real = torch.sum(real2, dim=0) / torch.sum(real2)
            kl += compute_kl(real, dist)
        else:
            dist = torch.sum(fakex, dim=0) / torch.sum(fakex)
            real = torch.sum(realx, dim=0) / torch.sum(realx)
            kl += compute_kl(real, dist)
    return kl

def train_gan(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes=100, is_dp=False):
    # Apply module validation
    generator = ModuleValidator.fix(generator)
    discriminator = ModuleValidator.fix(discriminator)

    # Ensure eval_frequency is set
    args.setdefault('eval_frequency', 10)

    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args['lr'])
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args['lr'])

    # Apply DP if enabled
    if is_dp:
        privacy_engine = PrivacyEngine()
        generator, g_optimizer, train_loader = privacy_engine.make_private(
            module=generator,
            optimizer=g_optimizer,
            data_loader=train_loader,
            noise_multiplier=args.get('noise_multiplier', 1.0),
            max_grad_norm=args['max_grad_norm'],
        )

    best_metrics = None
    best_model_state = None

    for epoch in range(args['epochs']):
        # Dynamic clipping bound decay
        current_clip_bound = args['max_grad_norm'] * (args['clip_decay_rate'] ** epoch)

        for i, data in enumerate(train_loader):
            if i >= itertimes:
                break

            real_data = data[0].to(args['device'])
            noise = torch.randn(args['batch_size'], args['z_dim'], device=args['device'])
            fake_data = generator(noise)

            # Train discriminator
            d_optimizer.zero_grad()
            d_loss_real = discriminator(real_data).mean()
            d_loss_fake = discriminator(fake_data.detach()).mean()
            d_loss = -d_loss_real + d_loss_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=current_clip_bound)
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            g_loss = -discriminator(fake_data).mean()
            kl_loss = KL_Loss(fake_data, real_data, col_type, col_dim)
            g_loss += args['kl_weight'] * kl_loss
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=current_clip_bound)
            g_optimizer.step()

        # Evaluate model at regular intervals
        if epoch % args['eval_frequency'] == 0:
            try:
                metrics = evaluate_model(generator, test_loader)  # Fixed function call
                print(f"Epoch {epoch}: Metrics = {metrics}")
                best_metrics, best_model_state = save_best_model(generator, metrics, best_metrics, best_model_state)
            except Exception as e:
                print(f"Error during evaluation at epoch {epoch}: {e}")

    # Save the final model
    torch.save(generator.state_dict(), 'final_model.pth')

# Wrapper functions for different training methods
def V_Train(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes=100):
    train_gan(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes, is_dp=False)

def W_Train(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes=100):
    train_gan(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes, is_dp=False)

def C_Train(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes=100):
    train_gan(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes, is_dp=False)

def C_Train_nofair(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes=100):
    train_gan(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes, is_dp=False)

def C_Train_dp(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes=100):
    train_gan(args, generator, discriminator, train_loader, test_loader, col_type, col_dim, itertimes, is_dp=True)
