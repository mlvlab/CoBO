import torch
import math
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import functools
from scipy import stats
import argparse

def lipschitz_loss(z, y, recon_weight):
    z = z.cuda()
    y = y.cuda()
    dif_y = (y.unsqueeze(1) - y.unsqueeze(0)).squeeze(-1)
    dif_z = torch.sqrt(torch.sum((z.unsqueeze(1) - z.unsqueeze(0))**2, dim=2) + 1e-10)
    lips = abs(dif_y / (dif_z + 1e-10))
    
    ratio = lips - torch.median(lips)
    ratio = (ratio * (recon_weight * recon_weight[:,None]).pow(0.5))
    ratio = ratio[ratio>0]
    loss = ratio.mean()

    return loss, torch.mean(lips), torch.mean(abs(dif_y)), dif_z.mean()

def update_models_end_to_end(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    learning_rte,
    num_update_epochs,
    track_with_wandb,
    tracker,
    alpha,
    beta,
    gamma,
    delta
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    objective.vae.train()
    model.train() 
    optimizer = torch.optim.Adam([
            {'params': objective.vae.parameters()},
            {'params': model.parameters()}], lr=learning_rte)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560/max_string_length)) 
    num_batches = math.ceil(len(train_x) / bsz)
    
    for _ in range(num_update_epochs):
        vae_losses = 0
        lip_losses = 0
        surr_losses = 0
        lipss = 0 
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            z, _, recon_loss, kldiv = objective.vae_forward(batch_list) 

            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float().cuda()
            pred = model(z)
            surr_loss = -mll(pred, batch_y.cuda()) 
                                
            data_weighter = DataWeighter()
            batch_y = batch_y.cpu().numpy()
            recon_weight = DataWeighter.normalize_weights(data_weighter.weighting_function(batch_y))
            batch_y = torch.from_numpy(batch_y).cuda()

            recon_weight = torch.from_numpy(recon_weight).cuda()
            recon_loss = recon_loss.mean([i for i in range(1, len(recon_loss.shape))])
            recon_loss = (recon_loss * recon_weight).mean()
            
            vae_loss = recon_loss + 0.1 * kldiv      
            lip_loss, lips, dif_y, dif_z = lipschitz_loss(z, batch_y, recon_weight)

            dim = z.shape[-1]
            c = math.exp(math.lgamma((dim+1)/2) - math.lgamma(dim/2))*2

            loss = alpha * lip_loss + beta * surr_loss + gamma * vae_loss + delta * (dif_z - c).abs()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
            optimizer.step()

            vae_losses += vae_loss.item() / num_batches
            lip_losses += lip_loss.item() / num_batches
            surr_losses += surr_loss.item() / num_batches
            lipss += lips.item() / num_batches

        if track_with_wandb:
            dict_log = {
                'vae_loss' : vae_losses,
                'lip_loss' : lip_losses,
                'surr_loss' : surr_losses,
                'lips' : lipss,
                'dif_y' : dif_y,
                'dif_z' : dif_z
            }
            tracker.log(dict_log)

    objective.vae.eval()
    model.eval()

    return objective, model

def update_surr_model(
    model,
    mll,
    learning_rte,
    train_z,
    train_y,
    n_epochs
):
    model = model.train()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte}], lr=learning_rte)
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model

class DataWeighter:
    def __init__(self, quantiles=0.95, noises=0.1):
        self.weighting_function = functools.partial(
            DataWeighter.dbas_weights,
            quantile=quantiles,
            noise=noises,
        )

    @staticmethod
    def normalize_weights(weights: np.array):
        """ Normalizes the given weights """
        return weights / np.mean(weights)

    @staticmethod
    def dbas_weights(properties: np.array, quantile: float, noise: float):
        y_star = np.quantile(properties, quantile)
        if np.isclose(noise, 0):
            weights = (properties >= y_star).astype(float)
        else:
            weights = stats.norm.sf(y_star, loc=properties, scale=noise)
        return weights