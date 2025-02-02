import os
import torch
import numpy as np
import pandas as pd
import random
import math
from torch import optim
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import torch.nn as nn
import torch.autograd as autograd

# compute kl loss (not use now)
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


# Utility function for dynamic clipping bound decay
def calculate_clipping_bound(strategy, i, C, C_prime, E, d):
    """
    Calculate the clipping bound based on the specified decay strategy.

    Args:
        strategy: Decay strategy ('none', 'linear', 'exponential', 'logarithmic').
        i: Current iteration or epoch.
        C: Initial clipping bound.
        C_prime: Final clipping bound (for linear decay).
        E: Total iterations or epochs.
        d: Decay factor (for exponential decay).

    Returns:
        Updated clipping bound.
    """
    if strategy == 'none':
        return C
    elif strategy == 'linear':
        return C - i * (C - C_prime) / (E - 1)
    elif strategy == 'exponential':
        return C * (d ** (i - 1))
    elif strategy == 'logarithmic':
        return C / (1 + math.log(i))
    else:
        raise ValueError(f"Invalid decay strategy: {strategy}")

# Preserving original GAN training methods: V_Train, C_Train, C_Train_nofair, C_Train_dp, W_Train
# Adding model selection and dynamic clipping functionality while retaining existing methods

def V_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, itertimes=100,
            steps_per_epoch=None, GPU=False, KL=True):
    """
    The vanilla (basic) training process for GAN
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sample_rows: # of synthesized rows
        * G: the generator
        * D: the discriminator
        * epochs: # of epochs
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
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

    # the default # of steps is the # of batches.
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

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

                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z)

                y_real = D(x_real)
                y_fake = D(x_fake)

                # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                # Avoid the suppress of Discriminator over Generator
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()

                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2

                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()

                ''' train Generator '''
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
                    # if GPU:
                    #    G.cpu()
                    #    G.GPU = False
                    for time in range(sample_times):
                        sample_data = None
                        for x_real in sampleloader:
                            z = torch.randn(x_real.shape[0], z_dim)
                            if GPU:
                                z = z.cuda()
                            x_fake = G(z)
                            samples = x_fake
                            samples = samples.reshape(samples.shape[0], -1)
                            samples = samples[:, :dataset.dim]
                            samples = samples.cpu()
                            sample_table = dataset.reverse(samples.detach().numpy())
                            df = pd.DataFrame(sample_table, columns=dataset.columns)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = pd.concat([sample_data, df], ignore_index=True)
                        sample_data.to_csv(path + 'sample_data_{}_{}_{}.csv'.format(t, epoch, time), index=None)
                    # if GPU:
                    #     G.cuda()
                    #    G.GPU = True
                    G.train()
                    break
    return G, D


def W_Train(t, path, sampleloader, G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, sample_times,
            itertimes=100, GPU=False, KL=True):
    """
    The WGAN training process for GAN
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sample_rows: # of synthesized rows
        * G: the generator
        * D: the discriminator
        * ng:
        * nd:
        * cp:
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
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

    epoch_time = int(ng / 100)
    # the default # of steps is the # of batches.

    for t1 in range(ng):
        for t2 in range(nd):
            x_real = dataloader.sample(dataloader.batch_size)
            if GPU:
                x_real = x_real.cuda()

            ''' train Discriminator '''
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
            D_optim.step()
            for p in D.parameters():
                p.data.clamp_(-cp, cp)  # clip the discriminator parameters (wgan)

        ''' train Generator '''
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
        if t1 % epoch_time == 0 and t1 > 0:
            G.eval()
            # if GPU:
            #    G.cpu()
            #    G.GPU = False
            for time in range(sample_times):
                sample_data = None
                for x_real in sampleloader:
                    z = torch.randn(x_real.shape[0], z_dim)
                    if GPU:
                        z = z.cuda()
                    x_fake = G(z)
                    samples = x_fake
                    samples = samples.reshape(samples.shape[0], -1)
                    samples = samples[:, :dataset.dim]
                    samples = samples.cpu()
                    sample_table = dataset.reverse(samples.detach().numpy())
                    df = pd.DataFrame(sample_table, columns=dataset.columns)
                    if sample_data is None:
                        sample_data = df
                    else:
                        sample_data = sample_data.append(df)
                sample_data.to_csv(path + 'sample_data_{}_{}_{}.csv'.format(t, int(t1 / epoch_time), time), index=None)
            # if GPU:
            #    G.cuda()
            #    G.GPU = True
            G.train()
    return G, D


def C_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, itertimes=100,
            steps_per_epoch=None, GPU=False):
    """
    The
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader:
    :param G:
    :param D:
    :param epochs:
    :param lr:
    :param dataloader:
    :param z_dim:
    :param dataset:
    :param itertimes:
    :param steps_per_epoch:
    :param GPU:
    :return:
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
            ''' train Discriminator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()

            x_fake = G(z, c_real)
            y_real = D(x_real, c_real)
            y_fake = D(x_fake, c_real)

            # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
            fake_label = torch.zeros(y_fake.shape[0], 1)
            real_label = np.ones([y_real.shape[0], 1])
            real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
            real_label = torch.from_numpy(real_label).float()
            if GPU:
                fake_label = fake_label.cuda()
                real_label = real_label.cuda()

            D_Loss1 = F.binary_cross_entropy(y_real, real_label)
            D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
            D_Loss = D_Loss1 + D_Loss2

            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()
            ''' train Generator '''
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

        G.eval()
        # if GPU:
        #    G.cpu()
        #    G.GPU = False
        for time in range(sample_times):
            sample_data = None
            for x, y in sampleloader:
                z = torch.randn(x.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    y = y.cuda()
                x_fake = G(z, y)
                x_fake = torch.cat((x_fake, y), dim=1)
                samples = x_fake
                samples = samples.reshape(samples.shape[0], -1)
                samples = samples[:, :dataset.dim]
                samples = samples.cpu()
                sample_table = dataset.reverse(samples.detach().numpy())
                df = pd.DataFrame(sample_table, columns=dataset.columns)
                if sample_data is None:
                    sample_data = df
                else:
                    sample_data = sample_data.append(df)
            sample_data.to_csv(path + 'sample_data_{}_{}_{}.csv'.format(t, epoch, time), index=None)
        # if GPU:
        #    G.cuda()
        #    G.GPU = True
        G.train()
    return G, D


def C_Train_nofair(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times,
                   itertimes=100, steps_per_epoch=None, GPU=False):
    """
    The
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader:
    :param G:
    :param D:
    :param epochs:
    :param lr:
    :param dataloader:
    :param z_dim:
    :param dataset:
    :param itertimes:
    :param steps_per_epoch:
    :param GPU:
    :return:
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
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                x_fake = G(z, c_real)
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()

                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2

                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()
                ''' train Generator '''
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
                    # if GPU:
                    #    G.cpu()
                    #    G.GPU = False
                    for time in range(sample_times):
                        sample_data = None
                        for x, y in sampleloader:
                            z = torch.randn(x.shape[0], z_dim)
                            if GPU:
                                z = z.cuda()
                                y = y.cuda()
                            x_fake = G(z, y)
                            x_fake = torch.cat((x_fake, y), dim=1)
                            samples = x_fake
                            samples = samples.reshape(samples.shape[0], -1)
                            samples = samples[:, :dataset.dim]
                            samples = samples.cpu()
                            sample_table = dataset.reverse(samples.detach().numpy())
                            df = pd.DataFrame(sample_table, columns=dataset.columns)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = sample_data.append(df)
                        sample_data.to_csv(path + 'sample_data_{}_{}_{}.csv'.format(t, epoch, time), index=None)
                    # if GPU:
                    #    G.cuda()
                    #    G.GPU = True
                    G.train()
                    break
    return G, D


def C_Train_dp(
    t, path, sampleloader, G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, eps,
    sample_times, itertimes=100, GPU=False, delta=0.00001, decay_strategy='none',
    cp_start=1.0, decay_factor=0.99
):
    """
    Conditional Training with Differential Privacy
    Args:
        t: The t-th training iteration.
        path: Path for storing logs.
        sampleloader: Data loader for sampling.
        G: Generator model.
        D: Discriminator model.
        ng: Number of generator steps.
        nd: Number of discriminator steps.
        cp: Initial clipping bound.
        lr: Learning rate.
        dataloader: Training data loader.
        z_dim: Dimension of the latent space.
        dataset: Dataset for reversible transformations.
        col_type: Column types for the dataset.
        eps: Epsilon for DP noise.
        sample_times: Number of samples to generate for evaluation.
        itertimes: Number of iterations (epochs) to train.
        GPU: Whether to use GPU.
        delta: Delta for DP noise.
        decay_strategy: Decay strategy for clipping bounds ('none', 'linear', 'exponential', 'logarithmic').
        cp_start: Initial clipping bound value.
        decay_factor: Decay factor for exponential clipping bound decay.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)

    def calculate_clipping_bound(strategy, i, C, C_prime, E, d):
        if strategy == 'none':
            return C
        elif strategy == 'linear':
            # Linear decay formula γ(i) = C - i * (C - C') / (E - 1)
            return C - i * (C - C_prime) / (E - 1)
        elif strategy == 'exponential':
            return C * (d ** (i - 1))
        elif strategy == 'logarithmic':
            return C / (1 + math.log(i))
        else:
            raise ValueError(f"Invalid decay strategy: {strategy}")

    q = dataloader.batch_size / len(dataloader.dataset)
    theta_n = 2 * q * math.sqrt(nd * math.log(1 / delta)) / eps

    max_epochs = 10  # Limit training to a maximum of 10 epochs

    # Training loop
    for t1 in range(1, min(itertimes + 1, max_epochs + 1)):
        # Decay clipping bound if needed
        cp = calculate_clipping_bound(decay_strategy, t1, cp_start, cp, itertimes, decay_factor)
        print(f"Epoch {t1}/{min(itertimes, max_epochs)}, Clipping Bound: {cp:.4f}")

        # Train Discriminator
        for t2 in range(nd):
            x_real, c_real = dataloader.sample(label=list(range(sample_times)))
            z = torch.randn(x_real.size(0), z_dim)

            if GPU:
                x_real, c_real, z = x_real.cuda(), c_real.cuda(), z.cuda()

            x_fake = G(z, c_real)
            y_real = D(x_real, c_real)
            y_fake = D(x_fake.detach(), c_real)

            # Compute loss
            D_Loss1 = -torch.mean(y_real)
            D_Loss2 = torch.mean(y_fake)
            D_Loss = D_Loss1 + D_Loss2

            # Add DP noise and clip gradients
            D_optimizer.zero_grad()
            D_Loss.backward()

            with torch.no_grad():  # Avoid unnecessary computation
                for p in D.parameters():
                    noise = torch.normal(0, theta_n, size=p.grad.shape).to(p.grad.device)
                    p.grad += noise
                    p.grad.data.clamp_(-cp, cp)

            # Optimization step
            D_optimizer.step()

        # Train Generator
        for t3 in range(ng):
            z = torch.randn(sample_times, z_dim)
            c_fake = torch.randint(0, len(dataset), (sample_times,))

            if GPU:
                z, c_fake = z.cuda(), c_fake.cuda()

            x_fake = G(z, c_fake)
            y_fake = D(x_fake, c_fake)

            # Compute loss
            G_Loss = -torch.mean(y_fake)

            # Backpropagation and optimization
            G_optimizer.zero_grad()
            G_Loss.backward()
            G_optimizer.step()

        # Log progress
        if t1 % 10 == 0 or t1 == 1:  # Log only every 10th epoch or the first
            print(f"Epoch {t1}/{min(itertimes, max_epochs)}, D Loss: {D_Loss.item():.4f}, G Loss: {G_Loss.item():.4f}")

        # Check for stopping condition
        if t1 >= max_epochs:
            print("Maximum number of epochs reached. Stopping training.")
            break

    # Generate synthetic samples after training
    z = torch.randn(sample_times, z_dim)
    c_fake = torch.randint(0, len(dataset), (sample_times,))

    if GPU:
        z, c_fake = z.cuda(), c_fake.cuda()

    synthetic_samples = G(z, c_fake).cpu().detach().numpy()
    print("Synthetic samples generated.")

    # Save the models
    torch.save(G.state_dict(), os.path.join(path, f"G_epoch_{t}.pth"))
    torch.save(D.state_dict(), os.path.join(path, f"D_epoch_{t}.pth"))
    return synthetic_samples
    pass


# Common Utility Functions

def calculate_frequencies(dataloader, dataset):
    """Calculate frequency distribution for real data."""
    real_data = np.concatenate([x.numpy() for x, _ in dataloader], axis=0)
    return np.histogramdd(real_data, bins=dataset.bins, range=dataset.ranges, density=True)[0]

def generate_synthetic_frequencies(G, dataloader, dataset, col_type):
    """Generate frequency distribution for synthetic data."""
    G.eval()
    synthetic_data = []
    for x, y in dataloader:
        z = torch.randn(x.size(0), G.z_dim)
        if G.GPU:
            z, y = z.cuda(), y.cuda()
        x_fake = G(z, y).cpu().detach().numpy()
        synthetic_data.append(dataset.reverse(x_fake))  # Reverse dataset transformations
    synthetic_data = np.concatenate(synthetic_data, axis=0)
    return np.histogramdd(synthetic_data, bins=dataset.bins, range=dataset.ranges, density=True)[0]

def calculate_overlap_ratio(real_freq, synthetic_freq):
    """Calculate overlap ratio between real and synthetic frequencies."""
    overlap = np.minimum(real_freq, synthetic_freq).sum()
    return overlap / real_freq.sum()

def JS_divergence(p, q):
    """Calculate the Jensen-Shannon Divergence between two distributions p and q."""
    p, q = np.asarray(p), np.asarray(q)
    p /= p.sum()  # Normalize to sum to 1
    q /= q.sum()
    return jensenshannon(p, q) ** 2

def Wasserstein_distance(p, q):
    """Compute the Wasserstein distance (1st order) between two distributions p and q."""
    p, q = np.asarray(p), np.asarray(q)
    return wasserstein_distance(p, q)

def adaptive_scoring(overlap_ratio, real_data_frequencies, synthetic_data_frequencies, scoring_weights):
    """Dynamically adjust the scoring metrics based on overlap ratio."""
    if overlap_ratio > 0.7:
        js_div = JS_divergence(real_data_frequencies, synthetic_data_frequencies)
        score = scoring_weights[0] * js_div
    else:
        wasser_dist = Wasserstein_distance(real_data_frequencies, synthetic_data_frequencies)
        score = scoring_weights[1] * wasser_dist
    return score, js_div if overlap_ratio > 0.7 else None, wasser_dist if overlap_ratio <= 0.7 else None

# Model Training Function

def train_gan_model(G, D, dataloader, z_dim, dataset, col_type, lr, eps, ng, nd, cp, sample_times, decay_strategy, cp_start, decay_factor, itertimes=100, GPU=False):
    """Helper function to train a GAN model (G, D) with dynamic clipping bound decay."""
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)

    # Clipping bound decay calculation
    def calculate_clipping_bound(i, C, C_prime, E, d):
        if decay_strategy == 'none':
            return C
        elif decay_strategy == 'linear':
            return C - i * (C - C_prime) / (E - 1)
        elif decay_strategy == 'exponential':
            return C * (d ** (i - 1))
        elif decay_strategy == 'logarithmic':
            return C / (1 + math.log(i))
        else:
            raise ValueError(f"Invalid decay strategy: {decay_strategy}")

    max_epochs = 10  # Limit training to a maximum of 10 epochs
    for epoch in range(1, min(itertimes + 1, max_epochs + 1)):
        cp = calculate_clipping_bound(epoch, cp_start, cp, itertimes, decay_factor)
        print(f"Epoch {epoch}/{min(itertimes, max_epochs)}, Clipping Bound: {cp:.4f}")

        # Discriminator training loop
        for _ in range(nd):
            x_real, c_real = dataloader.sample(label=list(range(sample_times)))
            z = torch.randn(x_real.size(0), z_dim)

            if GPU:
                x_real, c_real, z = x_real.cuda(), c_real.cuda(), z.cuda()

            x_fake = G(z, c_real)
            y_real = D(x_real, c_real)
            y_fake = D(x_fake.detach(), c_real)

            D_Loss = -torch.mean(y_real) + torch.mean(y_fake)
            D_optimizer.zero_grad()
            D_Loss.backward()

            # DP noise addition and clipping
            with torch.no_grad():
                for p in D.parameters():
                    noise = torch.normal(0, eps, size=p.grad.shape).to(p.grad.device)
                    p.grad += noise
                    p.grad.data.clamp_(-cp, cp)

            D_optimizer.step()

        # Generator training loop
        for _ in range(ng):
            z = torch.randn(sample_times, z_dim)
            c_fake = torch.randint(0, len(dataset), (sample_times,))

            if GPU:
                z, c_fake = z.cuda(), c_fake.cuda()

            x_fake = G(z, c_fake)
            y_fake = D(x_fake, c_fake)

            G_Loss = -torch.mean(y_fake)
            G_optimizer.zero_grad()
            G_Loss.backward()
            G_optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{min(itertimes, max_epochs)}, D Loss: {D_Loss.item():.4f}, G Loss: {G_Loss.item():.4f}")

    # Generate synthetic samples after training
    z = torch.randn(sample_times, z_dim)
    c_fake = torch.randint(0, len(dataset), (sample_times,))

    if GPU:
        z, c_fake = z.cuda(), c_fake.cuda()

    synthetic_samples = G(z, c_fake).cpu().detach().numpy()
    print("Synthetic samples generated.")

    return synthetic_samples

# Model Evaluation Function

def evaluate_models(model_pool, dataloader, dataset, col_type, scoring_weights=(0.5, 0.5)):
    """Evaluate multiple GAN models and select the best one based on frequency-driven metrics."""
    real_data_frequencies = calculate_frequencies(dataloader, dataset)
    best_model_name = None
    best_score = float('inf')
    best_model_metrics = {}

    for model_name, (G, _) in model_pool.items():
        synthetic_data_frequencies = generate_synthetic_frequencies(G, dataloader, dataset, col_type)

        overlap_ratio = calculate_overlap_ratio(real_data_frequencies, synthetic_data_frequencies)
        score, js_div, wasser_dist = adaptive_scoring(overlap_ratio, real_data_frequencies, synthetic_data_frequencies, scoring_weights)

        if score < best_score:
            best_score = score
            best_model_name = model_name
            best_model_metrics = {
                "overlap_ratio": overlap_ratio,
                "score": score,
                "JS_divergence": js_div,
                "Wasserstein_distance": wasser_dist
            }

    return best_model_name, best_model_metrics

# Main Model Training and Selection

def train_and_select_best_model(path, dataloader, sampleloader, dataset, col_type, z_dim, epochs, lr, eps, sample_times, GPU=False):
    """
    Train multiple GAN models and select the best-performing model using frequency-driven evaluation.
    """
    model_pool = {}

    # Train various models
    model_pool["V_Train"] = train_gan_model(V_Train, dataloader, z_dim, dataset, col_type, lr, eps, 1, 1, 0.5, sample_times, 'none', 1.0, 0.99)
    model_pool["C_Train"] = train_gan_model(C_Train, dataloader, z_dim, dataset, col_type, lr, eps, 1, 1, 0.5, sample_times, 'linear', 1.0, 0.99)
    model_pool["C_Train_nofair"] = train_gan_model(C_Train_nofair, dataloader, z_dim, dataset, col_type, lr, eps, 1, 1, 0.5, sample_times, 'exponential', 1.0, 0.99)
    model_pool["C_Train_dp"] = train_gan_model(C_Train_dp, dataloader, z_dim, dataset, col_type, lr, eps, 1, 1, 0.5, sample_times, 'logarithmic', 1.0, 0.99)
    model_pool["W_Train"] = train_gan_model(W_Train, dataloader, z_dim, dataset, col_type, lr, eps, 1, 1, 0.5, sample_times, 'linear', 1.0, 0.8)

    # Evaluate models
    best_model, best_model_metrics = evaluate_models(model_pool, dataloader, dataset, col_type)

    # Save results
    save_path = os.path.join(path, f"Best_{best_model}")
    torch.save(model_pool[best_model][0].state_dict(), save_path)
    print(f"Best Model: {best_model} saved at {save_path}. Metrics: {best_model_metrics}")
