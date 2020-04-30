import numpy as np
import torch
import os
from .model_matching import matching_upd_j
from dataset.models import MVCNN, MVFC, MVFCG

import logging

# logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger('HFL.match')

def match_global(local_models, num_views, layer_num, args):
    for model in local_models:
        model.to('cpu')
    match_pattern = []
    global_shape = []
    mixed_weights = []
    layer_weight_list, layer_match_pattern, matched_width = extract_weights(local_models, 0, num_views, args.total_views)
    match_pattern.append(layer_match_pattern)
    global_shape.append(matched_width)
    for layer_idx in range(layer_num-2):
        logger.debug('processing layer: %d/%d',layer_idx+1,layer_num-1)
        mixed_weight, layer_match_pattern, matched_width = one_layer_matching(layer_weight_list, match_pattern[-1], global_shape[-1], args.match_iter)
        mixed_weight = recompute_matched_width(mixed_weight, layer_weight_list, layer_match_pattern, match_pattern[-1])
        layer_weight_list, layer_match_pattern, matched_width = extract_weights(local_models, layer_idx+1, layer_match_pattern, matched_width)
        mixed_weights.append(mixed_weight)
        global_shape.append(matched_width)
        match_pattern.append(layer_match_pattern)
    logger.debug('processing layer: %d/%d',layer_num-1,layer_num-1)
    mixed_weight, layer_match_pattern, matched_width = last_layer_matching(layer_weight_list, match_pattern[-1], global_shape[-1])
    mixed_weights.append(mixed_weight)
    global_shape.append(matched_width)
    match_pattern.append(layer_match_pattern)
    for model in local_models:
        model.to(args.device)
    return mixed_weights, global_shape, match_pattern

def average_global(local_models, num_views, layer_num, args, global_shape, match_pattern):
    for model in local_models:
        model.to('cpu')
    mixed_weights = []
    layer_weight_list, layer_match_pattern, matched_width = extract_weights(local_models, 0, num_views, args.total_views)
    for layer_idx in range(layer_num-2):
        logger.debug('processing layer: %d/%d',layer_idx+1,layer_num-1)
        mixed_weight = recompute_matched_width(np.zeros((global_shape[layer_idx+1],global_shape[layer_idx]+1)), layer_weight_list, match_pattern[layer_idx+1], match_pattern[layer_idx])
        layer_weight_list, layer_match_pattern, matched_width = extract_weights(local_models, layer_idx+1, match_pattern[layer_idx+1], global_shape[layer_idx+1])
        mixed_weights.append(mixed_weight)
    logger.debug('processing layer: %d/%d',layer_num-1,layer_num-1)
    mixed_weight, layer_match_pattern, matched_width = last_layer_matching(layer_weight_list, match_pattern[-2], global_shape[-2])
    mixed_weights.append(mixed_weight)
    for model in local_models:
        model.to(args.device)
    return mixed_weights, global_shape, match_pattern

def extract_weights(local_models, layer, matching_patterns, total_length):
    padded_weights = []
    matching_pattern_new = []
    full_length = 0
    for model,pattern in zip(local_models,matching_patterns):
        net_weights = []
        state_dict = model.state_dict()
        for param_id, (k,v) in enumerate(state_dict.items()):
            if (layer*2 != param_id and layer*2+1 != param_id):
                continue
            if 'weight' in k:
                net_weights.append(v.numpy())
            else:
                net_weights.append(v.numpy().reshape(-1))
        if (layer == 0):
            block_size = net_weights[0].shape[1]/len(pattern)
            idx = []
            for i in pattern:
                idx.extend(np.arange(i*block_size,(i+1)*block_size))
            idx = np.array(idx)
        else:
            block_size = 1
            idx = np.array(pattern)
        idx = idx.astype(int)
        full_length = int(block_size*total_length)
        temp_weights = np.zeros((net_weights[0].shape[0], full_length + 1))
        temp_weights[:,idx] = net_weights[0]
        temp_weights[:,-1] = net_weights[1]
        padded_weights.append(temp_weights)
        matching_pattern_new.append(idx)
    return padded_weights,matching_pattern_new, full_length

def last_layer_matching(weight_list, match_patterns, last_layer_shape):
    this_layer_shape = weight_list[0].shape[0]
    mixed_weights = np.sum(weight_list, axis=0)
    counts = np.zeros(last_layer_shape+1)
    for pattern in match_patterns:
        counts[-1] += 1
        for idx in pattern:
            counts[idx]+=1
    avg_weights = np.diagflat(1/counts)
    mixed_weights = np.matmul(mixed_weights,avg_weights)
    match_pattern = [np.arange(this_layer_shape) for i in range(last_layer_shape)]
    return mixed_weights, match_pattern, this_layer_shape

def one_layer_matching(weight_list, old_matching_patterns, last_layer_length, iternum):
    sigma = 1.0
    gamma = 7.0
    mu0 =0.0
    mu0_b = 0.1

    J = len(weight_list)
    list_order = sorted(range(J), key=lambda x: -weight_list[x].shape[0])

    inv_sigma_layer = [[(len(old_matching_patterns[j]) * 1 / sigma)] for j in range(J)]
    inv_sigma_prior = np.array([(weight_list[0].shape[1] * 1 / sigma)])

    normalized_weights = np.array([w * s for w, s in zip(weight_list, inv_sigma_layer)])
    normalized_prior_mean = np.array([(mu0_b + mu0* (weight_list[0].shape[1]-1))/inv_sigma_prior])
    
    for j in range(J):
        temp_sigma = np.zeros(weight_list[j].shape[1])
        temp_sigma[old_matching_patterns[j]] = inv_sigma_layer[j]
        inv_sigma_layer[j] = temp_sigma
    logger.debug('initializing match settings')

    mixed_weights = normalized_prior_mean + normalized_weights[list_order[0]]
    mixed_sigmas = np.outer(np.ones(mixed_weights.shape[0]), inv_sigma_prior + inv_sigma_layer[list_order[0]])

    popularity_counts = [1] * mixed_weights.shape[0]

    match_patterns = [[] for _ in range(J)]
    match_patterns[list_order[0]] = list(range(mixed_weights.shape[0]))

    # normalized_weights = torch.from_numpy(normalized_weights).to(device)

    for j in list_order[1:]:
        mixed_weights, mixed_sigmas, popularity_counts, match_pattern_j = matching_upd_j(normalized_weights[j], mixed_weights, inv_sigma_layer[j], mixed_sigmas, normalized_prior_mean, inv_sigma_prior, popularity_counts, gamma, J)
        match_patterns[j] = match_pattern_j
    
    for itr in range(iternum):
        logger.debug('matching_iteration: %d/%d',itr+1,iternum)
        random_order = np.random.permutation(J)
        for j in random_order:
            to_delete = []
            ## Remove j
            Lj = len(match_patterns[j])
            for l, i in sorted(zip(range(Lj), match_patterns[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(match_patterns[j_clean]):
                            if i < l_ind and j_clean != j:
                                match_patterns[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                logger.warning('Warning - weird unmatching')
                else:
                    mixed_weights[i] = mixed_weights[i] - normalized_weights[j][l]
                    mixed_sigmas[i] -= inv_sigma_layer[j]

            mixed_weights = np.delete(mixed_weights, to_delete, axis=0)
            mixed_sigmas = np.delete(mixed_sigmas, to_delete, axis=0)
            mixed_weights, mixed_sigmas, popularity_counts, match_pattern_j = matching_upd_j(normalized_weights[j], mixed_weights, inv_sigma_layer[j], mixed_sigmas, normalized_prior_mean, inv_sigma_prior, popularity_counts, gamma, J)
            match_patterns[j] = match_pattern_j
    

    matched_width = mixed_weights.shape[0]
    mixed_weights = mixed_weights/mixed_sigmas

    return mixed_weights, match_patterns, matched_width

def build_local_models(mixed_weights, global_shape, matching_patterns, local_models):
    with torch.no_grad():
        layer_num = 0
        for layer_pattern_list, layer_pattern_next, layer_weights in zip(matching_patterns[:-1], matching_patterns[1:], mixed_weights):
            for local_model, local_pattern_this, local_pattern_next in zip(local_models, layer_pattern_list,layer_pattern_next):
                state_dict=local_model.state_dict()
                temp_layer_weights = layer_weights[local_pattern_next,:]
                local_weight = torch.from_numpy(temp_layer_weights[:,local_pattern_this])
                local_bias = torch.from_numpy(layer_weights[local_pattern_next, -1])
                for param_id, (k,v) in enumerate(state_dict.items()):
                    if (layer_num*2 == param_id):
                        v.copy_(local_weight)
                    elif(layer_num*2+1 == param_id):
                        v.copy_(local_bias.view(-1))
            layer_num += 1
    return local_models

def build_global_models(mixed_weights, global_shape, matching_patterns, args):
    global_model = MVFCG('global_model', args.num_classes, cnn_name = args.model, shape = global_shape)
    state_dict=global_model.state_dict()
    for param_id, (k,v) in enumerate(state_dict.items()):
        if (param_id%2 == 0):
            local_weight = torch.from_numpy(mixed_weights[param_id//2][:,:-1])
            v.copy_(local_weight)
        elif(param_id%2 == 1):
            local_bias = torch.from_numpy(mixed_weights[param_id//2][:,-1])
            v.copy_(local_bias.view(-1))
    return global_model

def recompute_matched_width(mixed_weight, layer_weight_list, layer_match_pattern, last_layer_pattern):
    matched_count = np.zeros_like(mixed_weight)
    match_pattern = np.zeros_like(mixed_weight)
    mixed_weight.fill(0)
    this_temp = np.arange(match_pattern.shape[0])
    last_temp = np.arange(match_pattern.shape[1])
    for local_weight, this_layer, last_layer in zip(layer_weight_list, layer_match_pattern, last_layer_pattern):
        match_pattern.fill(1)
        match_pattern[:,~np.isin(last_temp, last_layer)] = 0
        match_pattern[:,-1] = 1
        match_pattern[~np.isin(this_temp, this_layer),:] = 0
        matched_count += match_pattern
        mixed_weight[this_layer,:]+= local_weight
    matched_count[matched_count==0]=1
    mixed_weight /= matched_count
    return mixed_weight

