import numpy as np
import torch
from lapsolver import solve_dense
import logging
import time

# logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger('HFL.match')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J):

    L = global_weights.shape[0]
    compute_cost_start = time.time()
    full_cost = compute_cost(global_weights.astype(np.float32), weights_j.astype(np.float32), global_sigmas.astype(np.float32), sigma_inv_j.astype(np.float32), prior_mean_norm.astype(np.float32), prior_inv_sigma.astype(np.float32), popularity_counts, gamma, J)
    compute_cost_dur = time.time() - compute_cost_start
    # logger.debug('cost size: [%d,%d], time cost: %f',full_cost.shape[0],full_cost.shape[1], compute_cost_dur)
    row_ind, col_ind = solve_dense(-full_cost)
    # logger.debug('solved for matching pattern')
    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))
    total_dur = time.time() - compute_cost_start
    # logger.debug('new length: %d, time cost: %f',new_L, total_dur)
    return global_weights, global_sigmas, popularity_counts, assignment_j

def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J):
    
    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts, dtype=np.float32), 10)
    
    with torch.no_grad():
        global_weights_t = torch.from_numpy(global_weights).to(device)
        global_sigmas_t = torch.from_numpy(global_sigmas).to(device)
        sij_p_gs = torch.from_numpy(sigma_inv_j + global_sigmas).to(device)
        # sij_p_gs.pow_(0.5)
        red_term = global_weights_t.pow(2).div(global_sigmas_t).sum(dim=1)
        weights_j_t = torch.from_numpy(weights_j).to(device)
        # weights_j_t.div_(sij_p_gs)
        # global_weights_t.div_(sij_p_gs)
        # compute_cost_start = time.time()
        # para_cost = global_weights_t.add(weights_j_t).pow(2).sum(dim=2)
        para_cost = torch.stack([row_param_cost_simplified(global_weights_t, weights_j_t[l],sij_p_gs) for l in range(Lj)],dim=0)
        # print("para_cost.shape: ", para_cost.shape)
        # print("red_term.shape: ", red_term.shape)
        # compute_cost_dur = time.time()-compute_cost_start
        # logger.debug("loop time: %f", compute_cost_dur)
        param_cost = para_cost.add_(red_term, alpha = -1).to('cpu').numpy()
    param_cost += np.log(counts / (J - counts))
    # print("para_cost.shape: ", param_cost.shape)
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added, dtype=np.float32))
    # nonparam_cost = np.tile((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (prior_mean_norm ** 2 / prior_inv_sigma).sum()), (1,max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    del(weights_j_t)
    del(sij_p_gs)
    del(para_cost)
    del(global_weights_t)
    del(global_sigmas_t)
    
    full_cost = np.hstack((param_cost, nonparam_cost)).astype(np.float32)
    return full_cost

def row_param_cost_simplified(global_weights, weights_j_l,sigma):
    match_norms = global_weights.add(weights_j_l).pow(2).div(sigma).sum(dim=1)
    return match_norms