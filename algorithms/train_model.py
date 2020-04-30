import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import logging

# logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger('HFL.train')

def train_local(local_models, feature_extractors, num_views, dataloaders, train_index, args):
    iters = []
    for model in local_models:
        model.train()
    for model in feature_extractors:
        model.train()
    parameter_set = set()
    for idx in train_index:
        parameter_set |= set(local_models[idx].parameters())
        for idx_i in num_views[idx]:
            parameter_set |= set(feature_extractors[idx_i].parameters())
        iters.append(iter(dataloaders[idx]))
    
    optimizer = optim.SGD(parameter_set, lr = args.lr, momentum=0.5)
    # optimizer = optim.RMSprop(parameter_set, lr = args.lr, alpha= 0.9, momentum=0.1)
    criterion = nn.CrossEntropyLoss()
    losses = []
    total = 0
    correct = 0
    for local_iter in range(args.local_iter):    
        optimizer.zero_grad()
        for idx in train_index:
            # iterator = iter(dataloaders[idx])
            try:
                input_features, targets = iters[idx].next()
            except StopIteration:
                iters[idx] = iter(dataloaders[idx])
                input_features, targets = iters[idx].next()
            # input_features = input_features.to(args.device)
            # print(targets.shape, input_features[0].shape)
            targets = targets.to(args.device)
            total += targets.size(0)
            extracted_features = []
            for feature, view in zip(input_features, num_views[idx]):
                extracted_features.append(feature_extractors[view](feature.to(args.device)))
            concat_features=torch.cat(extracted_features, dim= 1)
            output = local_models[idx](concat_features)
            loss = criterion(output,targets)
            losses.append(loss.detach().to('cpu').numpy())
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == targets).sum().item()
            loss.backward()
        optimizer.step()
        logger.debug('itr: %d, avg_loss: %f, avg_acc: %f',local_iter,sum(losses)/len(losses),correct/total)
    return local_models, feature_extractors

def compute_local_accuracy(local_models, feature_extractors, num_views, dataloaders, test_index, args):
    accuracy = []
    loss = []
    criterion = nn.CrossEntropyLoss()
    for model in local_models:
        model.eval()
    for model in feature_extractors:
        model.eval()
    with torch.no_grad():
        for idx in test_index:
            local_loss = 0
            total = 0
            correct = 0
            # print(dataloaders[idx].__len__())
            for iter_num, (input_features, targets) in enumerate(dataloaders[idx]):
                # input_features = input_features.to(args.device)
                targets = targets.to(args.device)
                extracted_features = []
                for feature, view in zip(input_features, num_views[idx]):
                    extracted_features.append(feature_extractors[view](feature.to(args.device)))
                concat_features=torch.cat(extracted_features, dim= 1)
                output = local_models[idx](concat_features)
                local_loss += criterion(output,targets)
                total += targets.size(0)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == targets).sum().item()
            accuracy.append(correct / total)
            loss.append(local_loss/(iter_num+1))
            logger.debug('agent: %d, avg_loss: %f, avg_acc: %f',idx,loss[-1].item(),accuracy[-1])
    return accuracy,loss

def compute_global_accuracy(global_model, feature_extractors, num_views, dataloader, args):
    criterion = nn.CrossEntropyLoss()
    global_model.to(args.device)
    global_model.eval()
    for view in num_views:
        feature_extractors[view].eval()
    with torch.no_grad():
        local_loss = 0
        total = 0
        correct = 0
        iter_num = 0
        for input_features, targets in dataloader:
            # input_features = input_features.to(args.device)
            targets = targets.to(args.device)
            extracted_features = []
            for feature, view in zip(input_features, num_views):
                extracted_features.append(feature_extractors[view](feature.to(args.device)))
            concat_features=torch.cat(extracted_features, dim= 1)
            output = global_model(concat_features)
            local_loss += criterion(output,targets)
            total += targets.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == targets).sum().item()
            iter_num += 1
        accuracy=correct / total
        loss=local_loss/iter_num
    global_model.to('cpu')
    return accuracy,loss
