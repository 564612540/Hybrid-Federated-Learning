import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader


from dataset.models import MVCNN, MVFC, MVFCG
from dataset.dataloader import MultiviewImgDataset

from model_matching.combine_nets import build_global_models, build_local_models, match_global, average_global
from algorithms.train_model import train_local, compute_local_accuracy, compute_global_accuracy
from utils.utils import add_args, save_log

import logging

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger('HFL')
# logger.setLevel(logging.DEBUG)#filename='logger.log', 

with_test = True

if __name__ == "__main__":
    args = add_args(argparse.ArgumentParser(description='HybridFL'))

    global_dataset = DataLoader(MultiviewImgDataset(args.root+'/*/test', num_models=list(range(args.num_classes)), num_views=list(range(args.total_views))),batch_size = args.batch_size*4)
    local_dataset_train = []
    local_dataset_test = []
    feature_extractors = []
    local_models = []
    num_views = []
    num_models = []

    with open(args.views_dir,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            num_views.append(np.fromstring(line, dtype=int, sep=','))
    assert len(num_views) == args.num_agents, 'wrong number of views for agents'

    with open(args.classes_dir,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            num_models.append(np.fromstring(line, dtype=int, sep=','))
    assert len(num_models) == args.num_agents, 'wrong number of models for agents'

    for j in range(args.num_agents):
        local_dataset_train.append(DataLoader(MultiviewImgDataset(args.root+'/*/train', num_models=num_models[j], num_views=num_views[j]),batch_size = args.batch_size, shuffle=True))
        local_dataset_test.append(DataLoader(MultiviewImgDataset(args.root+'/*/test', num_models=num_models[j], num_views=num_views[j]),batch_size = args.batch_size*4))
        local_models.append(MVFC('LM'+str(j), nclasses=args.num_classes, cnn_name=args.model, num_views=num_views[j]).to(args.device))
    for i in range(args.total_views):
        feature_extractors.append(MVCNN('F'+str(i), nclasses=args.num_classes, cnn_name=args.model).to(args.device))

    train_index = list(range(args.num_agents))

    global_model = None
    global_shape = None
    match_pattern = None

    lr_schedule = []
    with open(args.decay,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            lr_schedule.append(np.fromstring(line, dtype=float, sep=','))
    assert len(lr_schedule[0]) == len(lr_schedule[1]), 'learning rate schedule incorrect'

    for global_iter in range(args.comm_rounds):
        logger.info('Global_iter: %d/%d',global_iter,args.comm_rounds)
        if global_iter in lr_schedule[0]:
            args.lr *= lr_schedule[1][np.where(lr_schedule[0]==global_iter)]
            args.lr = args.lr[0]
        else:
            args.lr *= 0.95
        local_models, feature_extractors = train_local(local_models, feature_extractors, num_views, local_dataset_train, train_index, args)
        logger.info('finished_local_training')
        if with_test:
            local_acc, local_loss = compute_local_accuracy(local_models, feature_extractors, num_views, local_dataset_test, train_index, args)
            logger.info('local_acc: %f',local_acc[0])
        if (global_iter < args.layer_num):
            mixed_weights, global_shape, match_pattern = match_global(local_models, num_views, args.layer_num, args)
        else:
            mixed_weights, global_shape, match_pattern = average_global(local_models, num_views, args.layer_num, args, global_shape, match_pattern)
        logger.info('finished matching')
        global_model = build_global_models(mixed_weights, global_shape, match_pattern, args)
        logger.info('finished constructing global network')
        global_acc, global_loss = compute_global_accuracy(global_model, feature_extractors, list(range(args.total_views)), global_dataset, args)
        logger.info('global_acc: %f',global_acc)
        local_models = build_local_models(mixed_weights, global_shape, match_pattern, local_models)
        logger.info('finished building local networks')
        if with_test:
            local_acc_post, local_loss_post = compute_local_accuracy(local_models, feature_extractors, num_views, local_dataset_train, train_index, args)
            logger.info('local_acc_post: %f',local_acc_post[0])
            save_log(global_iter, local_acc, local_loss, local_acc_post, local_loss_post, global_acc, global_loss, global_shape, args)
    
    for model in local_models:
        model.save(args.log_dir,epoch=args.comm_rounds)
    for model in feature_extractors:
        model.save(args.log_dir,epoch=args.comm_rounds)
    global_model.save(args.log_dir,epoch=args.comm_rounds)