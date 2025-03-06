import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import math
from scipy.stats import wasserstein_distance
from scipy.special import kl_div


# Compute KL loss
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
            dist = torch.sum(fake2, dim=0)
            dist = dist / torch.sum(dist)
            real = torch.sum(real2, dim=0)
            real = real / torch.sum(real)
            kl += compute_kl(real, dist)
        else:
            dist = torch.sum(fakex, dim=0)
            dist = dist / torch.sum(dist)
            real = torch.sum(realx, dim=0)
            real = real / torch.sum(real)
            kl += compute_kl(real, dist)
    return kl


def gradient_penalty(real_data, fake_data, discriminator, lambda_gp=10.0, device='cuda'):
    """
    Computes the gradient penalty for WGAN-GP.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_data)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def gradient_sanitization(gradients, noise_scale=0.01, clip_value=1.0):
    """
    Sanitizes gradients by adding noise and clipping.
    """
    noise = torch.randn_like(gradients) * noise_scale
    sanitized_gradients = gradients + noise
    sanitized_gradients = torch.clamp(sanitized_gradients, -clip_value, clip_value)
    return sanitized_gradients


def compute_frequency_distribution(data, col_type, col_dim):
    """
    Compute frequency distributions for each column based on its type.
    """
    distributions = []
    sta = 0  # Start index
    for i in range(len(col_type)):
        dim = col_dim[i]  # Dimension of the current column
        end = sta + dim  # End index
        column_data = data[:, sta:end]

        if col_type[i] == "binary" or col_type[i] == "one-hot":
            dist = torch.sum(column_data, dim=0) / column_data.shape[0]
        elif col_type[i] == "normalize" or col_type[i] == "gmm":
            dist, _ = torch.histogram(column_data, bins=10, range=(0, 1))
            dist = dist / torch.sum(dist)
        distributions.append(dist.detach())  # Detach the tensor before appending

        sta = end  # Update start index for the next column
    return distributions


def add_differential_noise(distributions, noise_scale=0.01):
    """
    Add Gaussian noise to frequency distributions for differential privacy.
    """
    noisy_distributions = []
    for dist in distributions:
        noise = torch.randn_like(dist) * noise_scale
        noisy_dist = dist + noise
        noisy_dist = torch.clamp(noisy_dist, 0, 1)  # Ensure valid probabilities
        noisy_dist = noisy_dist / torch.sum(noisy_dist)  # Renormalize
        noisy_distributions.append(noisy_dist.detach())  # Detach the tensor before appending
    return noisy_distributions


def compare_distributions(real_dist, fake_dist, metric="wasserstein"):
    """
    Compare real and synthetic frequency distributions using a divergence metric.
    """
    divergence = 0.0
    for r, f in zip(real_dist, fake_dist):
        if metric == "wasserstein":
            # Detach tensors and convert to NumPy arrays
            r_np = r.detach().cpu().numpy()
            f_np = f.detach().cpu().numpy()
            divergence += wasserstein_distance(r_np, f_np)
        elif metric == "kl_divergence":
            # Detach tensors and convert to NumPy arrays
            r_np = r.detach().cpu().numpy()
            f_np = f.detach().cpu().numpy()
            divergence += kl_div(r_np, f_np).sum()
    return divergence


def save_best_model(G, D, divergence, best_divergence, best_G_state, best_D_state, path):
    """
    Save the best model based on the divergence metric.
    """
    if divergence < best_divergence:
        best_divergence = divergence
        best_G_state = G.state_dict()
        best_D_state = D.state_dict()
        torch.save(best_G_state, os.path.join(path, "best_G.pth"))
        torch.save(best_D_state, os.path.join(path, "best_D.pth"))
    return best_divergence, best_G_state, best_D_state


def V_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, itertimes=100,
            steps_per_epoch=None, GPU=False, KL=True):
    """
    The vanilla (basic) training process for GAN with gradient penalty and frequency distribution comparison.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True

    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    best_divergence = float("inf")
    best_G_state = None
    best_D_state = None

    for epoch in range(epochs):
        it = 0
        log = open(path + "train_log_" + str(t) + ".txt", "a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        while it < steps_per_epoch:
            for x_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()

                ''' Train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z)

                y_real = D(x_real)
                y_fake = D(x_fake)

                # Gradient Penalty
                gp = gradient_penalty(x_real, x_fake, D, lambda_gp=10.0, device='cuda' if GPU else 'cpu')

                # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))  # WGAN loss
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()

                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2 + gp  # Add gradient penalty

                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()

                ''' Train Generator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z)
                y_fake = D(x_fake)

                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()

                G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
                if KL:
                    KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
                    G_Loss = G_Loss1 + KL_loss
                else:
                    G_Loss = G_Loss1

                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()

                it += 1

                if it % itertimes == 0:
                    log = open(path + "train_log_" + str(t) + ".txt", "a+")
                    log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))
                    log.close()
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))

                if it >= steps_per_epoch:
                    G.eval()
                    for time in range(sample_times):
                        sample_data = None
                        for x_real in sampleloader:
                            z = torch.randn(x_real.shape[0], z_dim)
                            if GPU:
                                z = z.cuda()
                            x_fake = G(z)
                            samples = x_fake.reshape(x_fake.shape[0], -1)
                            samples = samples[:, :dataset.dim]
                            samples = samples.cpu()
                            sample_table = dataset.reverse(samples.detach().numpy())
                            df = pd.DataFrame(sample_table, columns=dataset.columns)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = pd.concat([sample_data, df], ignore_index=True)
                        sample_data.to_csv(path + 'sample_data_{}_{}_{}.csv'.format(t, epoch, time), index=None)
                    G.train()
                    break

        # Frequency Distribution Comparison
        real_dist = compute_frequency_distribution(x_real, col_type, dataset.col_dim)
        noisy_real_dist = add_differential_noise(real_dist, noise_scale=0.01)
        fake_dist = compute_frequency_distribution(x_fake, col_type, dataset.col_dim)
        divergence = compare_distributions(noisy_real_dist, fake_dist, metric="wasserstein")

        # Save the best model
        best_divergence, best_G_state, best_D_state = save_best_model(
            G, D, divergence, best_divergence, best_G_state, best_D_state, path
        )

    return G, D


def W_Train(t, path, sampleloader, G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, sample_times,
            itertimes=100, GPU=False, KL=True):
    """
    The WGAN training process with gradient penalty, gradient sanitization, and frequency distribution comparison.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True

    D_optim = optim.RMSprop(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.RMSprop(G.parameters(), lr=lr, weight_decay=0.00001)

    best_divergence = float("inf")
    best_G_state = None
    best_D_state = None

    for t1 in range(ng):
        for t2 in range(nd):
            x_real = dataloader.sample(dataloader.batch_size)
            if GPU:
                x_real = x_real.cuda()

            ''' Train Discriminator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()

            x_fake = G(z)

            y_real = D(x_real)
            y_fake = D(x_fake)

            D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))

            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()

            # Sanitize discriminator gradients
            for p in D.parameters():
                if p.grad is not None:
                    p.grad = gradient_sanitization(p.grad, noise_scale=0.01, clip_value=cp)

            D_optim.step()

            for p in D.parameters():
                p.data.clamp_(-cp, cp)  # Clip the discriminator parameters (WGAN)

        ''' Train Generator '''
        z = torch.randn(dataloader.batch_size, z_dim)
        if GPU:
            z = z.cuda()

        x_fake = G(z)
        y_fake = D(x_fake)

        G_Loss1 = -torch.mean(y_fake)
        if KL:
            KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            G_Loss = G_Loss1 + KL_loss
        else:
            G_Loss = G_Loss1

        G_optim.zero_grad()
        D_optim.zero_grad()
        G_Loss.backward()

        # Sanitize generator gradients
        for p in G.parameters():
            if p.grad is not None:
                p.grad = gradient_sanitization(p.grad, noise_scale=0.01, clip_value=cp)

        G_optim.step()

        if t1 % itertimes == 0:
            print("------ng : {}-------".format(t1))
            print("generator loss: {}".format(G_Loss.data))
            print("discriminator loss: {}".format(D_Loss.data))
            log = open(path + "train_log_" + str(t) + ".txt", "a+")
            log.write("----------ng: {}---------\n".format(t1))
            log.write("generator loss: {}\n".format(G_Loss.data))
            log.write("discriminator loss: {}\n".format(D_Loss.data))
            log.close()

        if t1 % 100 == 0:  # Frequency Distribution Comparison every 100 iterations
            real_dist = compute_frequency_distribution(x_real, col_type, dataset.col_dim)
            noisy_real_dist = add_differential_noise(real_dist, noise_scale=0.01)
            fake_dist = compute_frequency_distribution(x_fake, col_type, dataset.col_dim)
            divergence = compare_distributions(noisy_real_dist, fake_dist, metric="wasserstein")

            # Save the best model
            best_divergence, best_G_state, best_D_state = save_best_model(
                G, D, divergence, best_divergence, best_G_state, best_D_state, path
            )

    return G, D


def C_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, itertimes=100,
            steps_per_epoch=None, GPU=False):
    """
    The conditional GAN training process with gradient penalty and frequency distribution comparison.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True

    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    best_divergence = float("inf")
    best_G_state = None
    best_D_state = None

    for epoch in range(epochs):
        log = open(path + "train_log_" + str(t) + ".txt", "a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        for it in range(steps_per_epoch):
            c = random.choice(conditions)
            x_real, c_real = dataloader.sample(label=list(c))
            if GPU:
                x_real = x_real.cuda()
                c_real = c_real.cuda()

            ''' Train Discriminator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()

            x_fake = G(z, c_real)
            y_real = D(x_real, c_real)
            y_fake = D(x_fake, c_real)

            # Gradient Penalty
            gp = gradient_penalty(x_real, x_fake, D, lambda_gp=10.0, device='cuda' if GPU else 'cpu')

            # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))  # WGAN loss
            fake_label = torch.zeros(y_fake.shape[0], 1)
            real_label = np.ones([y_real.shape[0], 1])
            real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
            real_label = torch.from_numpy(real_label).float()
            if GPU:
                fake_label = fake_label.cuda()
                real_label = real_label.cuda()

            D_Loss1 = F.binary_cross_entropy(y_real, real_label)
            D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
            D_Loss = D_Loss1 + D_Loss2 + gp  # Add gradient penalty

            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()

            ''' Train Generator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()

            x_fake = G(z, c_real)
            y_fake = D(x_fake, c_real)

            real_label = torch.ones(y_fake.shape[0], 1)
            if GPU:
                real_label = real_label.cuda()

            G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
            KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            G_Loss = G_Loss1 + KL_loss

            G_optim.zero_grad()
            D_optim.zero_grad()
            G_Loss.backward()
            G_optim.step()

            if it % itertimes == 0:
                log = open(path + "train_log_" + str(t) + ".txt", "a+")
                log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))
                log.close()
                print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))

        # Frequency Distribution Comparison
        real_dist = compute_frequency_distribution(x_real, col_type, dataset.col_dim)
        noisy_real_dist = add_differential_noise(real_dist, noise_scale=0.01)
        fake_dist = compute_frequency_distribution(x_fake, col_type, dataset.col_dim)
        divergence = compare_distributions(noisy_real_dist, fake_dist, metric="wasserstein")

        # Save the best model
        best_divergence, best_G_state, best_D_state = save_best_model(
            G, D, divergence, best_divergence, best_G_state, best_D_state, path
        )

    return G, D


def C_Train_nofair(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times,
                   itertimes=100, steps_per_epoch=None, GPU=False):
    """
    The conditional GAN training process without fairness constraints, with gradient penalty and frequency distribution comparison.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True

    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    best_divergence = float("inf")
    best_G_state = None
    best_D_state = None

    for epoch in range(epochs):
        log = open(path + "train_log_" + str(t) + ".txt", "a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        it = 0
        while it < steps_per_epoch:
            for x_real, c_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()

                ''' Train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z, c_real)
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)

                # Gradient Penalty
                gp = gradient_penalty(x_real, x_fake, D, lambda_gp=10.0, device='cuda' if GPU else 'cpu')

                # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))  # WGAN loss
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()

                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2 + gp  # Add gradient penalty

                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()

                ''' Train Generator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z, c_real)
                y_fake = D(x_fake, c_real)

                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()

                G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
                KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
                G_Loss = G_Loss1 + KL_loss

                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()

                it += 1

                if it % itertimes == 0:
                    log = open(path + "train_log_" + str(t) + ".txt", "a+")
                    log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))
                    log.close()
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it, D_Loss.data, G_Loss.data))

                if it >= steps_per_epoch:
                    G.eval()
                    for time in range(sample_times):
                        sample_data = None
                        for x, y in sampleloader:
                            z = torch.randn(x.shape[0], z_dim)
                            if GPU:
                                z = z.cuda()
                                y = y.cuda()
                            x_fake = G(z, y)
                            x_fake = torch.cat((x_fake, y), dim=1)
                            samples = x_fake.reshape(x_fake.shape[0], -1)
                            samples = samples[:, :dataset.dim]
                            samples = samples.cpu()
                            sample_table = dataset.reverse(samples.detach().numpy())
                            df = pd.DataFrame(sample_table, columns=dataset.columns)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = pd.concat([sample_data, df], ignore_index=True)
                        sample_data.to_csv(path + 'sample_data_{}_{}_{}.csv'.format(t, epoch, time), index=None)
                    G.train()
                    break

        # Frequency Distribution Comparison
        real_dist = compute_frequency_distribution(x_real, col_type, dataset.col_dim)
        noisy_real_dist = add_differential_noise(real_dist, noise_scale=0.01)
        fake_dist = compute_frequency_distribution(x_fake, col_type, dataset.col_dim)
        divergence = compare_distributions(noisy_real_dist, fake_dist, metric="wasserstein")

        # Save the best model
        best_divergence, best_G_state, best_D_state = save_best_model(
            G, D, divergence, best_divergence, best_G_state, best_D_state, path
        )

    return G, D


def C_Train_dp(t, path, sampleloader, G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, eps, sample_times,
               itertimes=100, GPU=False, delta=0.00001):
    """
    The Conditional Training with Differential Privacy, gradient sanitization, and frequency distribution comparison.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True

    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.RMSprop(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.RMSprop(G.parameters(), lr=lr, weight_decay=0.00001)

    q = dataloader.batch_size / len(dataloader.data)
    theta_n = 2 * q * math.sqrt(nd * math.log(1 / delta)) / eps
    epoch_time = int(ng / 5)
    print("theta_n: {}".format(theta_n))

    best_divergence = float("inf")
    best_G_state = None
    best_D_state = None

    for t1 in range(ng):
        for c in conditions:
            for t2 in range(nd):
                x_real, c_real = dataloader.sample(label=list(c))
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()

                ''' Train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z, c_real)
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)

                D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))

                D_optim.zero_grad()
                G_optim.zero_grad()
                D_Loss.backward()

                # Gradient Sanitization for Differential Privacy
                for p in D.parameters():
                    if p.grad is not None:
                        sigma = theta_n * 1
                        noise = np.random.normal(0, sigma, p.grad.shape) / dataloader.batch_size
                        noise = torch.from_numpy(noise).float()
                        if GPU:
                            noise = noise.cuda()
                        p.grad += noise

                D_optim.step()
                for p in D.parameters():
                    p.data.clamp_(-cp, cp)  # Clip the discriminator parameters (WGAN)

            ''' Train Generator '''
            x_real, c_real = dataloader.sample(label=list(c))
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
                c_real = c_real.cuda()

            x_fake = G(z, c_real)
            y_fake = D(x_fake, c_real)

            G_Loss = -torch.mean(y_fake)  # Fixed extra parenthesis here

            G_optim.zero_grad()
            D_optim.zero_grad()
            G_Loss.backward()

            # Gradient Sanitization for Generator
            for p in G.parameters():
                if p.grad is not None:
                    p.grad = gradient_sanitization(p.grad, noise_scale=0.01, clip_value=cp)

            G_optim.step()

        if t1 % itertimes == 0:
            print("------ng : {}-------".format(t1))
            print("generator loss: {}".format(G_Loss.data))
            print("discriminator loss: {}".format(torch.mean(D_Loss).data))
            log = open(path + "train_log_" + str(t) + ".txt", "a+")
            log.write("----------ng: {}---------\n".format(t1))
            log.write("generator loss: {}\n".format(G_Loss.data))
            log.write("discriminator loss: {}\n".format(torch.mean(D_Loss).data))
            log.close()

        if (t1 + 1) % epoch_time == 0 and t1 > 0:
            G.eval()
            if GPU:
                G.cpu()
                G.GPU = False
            for time in range(sample_times):
                y = torch.from_numpy(sampleloader.label).float()
                z = torch.randn(len(sampleloader.label), z_dim)
                if GPU:
                    z = z.cuda()
                    y = y.cuda()
                x_fake = G(z, y)
                x_fake = torch.cat((x_fake, y), dim=1)
                samples = x_fake.cpu()
                samples = samples.reshape(samples.shape[0], -1)
                samples = samples[:, :dataset.dim]
                sample_table = dataset.reverse(samples.detach().numpy())
                sample_data = pd.DataFrame(sample_table, columns=dataset.columns)
                sample_data.to_csv(
                    path + 'sample_data_{}_{}_{}_{}.csv'.format(eps, t, int(t1 / epoch_time), time),
                    index=None
                )
            if GPU:
                G.cuda()
                G.GPU = True
            G.train()

            # Frequency Distribution Comparison
        real_dist = compute_frequency_distribution(x_real, col_type, dataset.col_dim)
        noisy_real_dist = add_differential_noise(real_dist, noise_scale=0.01)
        fake_dist = compute_frequency_distribution(x_fake, col_type, dataset.col_dim)
        divergence = compare_distributions(noisy_real_dist, fake_dist, metric="wasserstein")

        # Save the best model
        best_divergence, best_G_state, best_D_state = save_best_model(
            G, D, divergence, best_divergence, best_G_state, best_D_state, path
        )

    return G, D