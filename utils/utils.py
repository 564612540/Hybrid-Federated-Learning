import torch
import os
import argparse

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='alexnet', metavar='N', help='neural network used in training')
    parser.add_argument('--total_views', type=int, default=12, metavar='N', help='total number of views')
    parser.add_argument('--num_agents', type=int, default=12, metavar='N', help='total number of agents')
    parser.add_argument('--num_classes', type=int, default=40, metavar='N', help='number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--comm_rounds', type=int, default=100, metavar='CR', help='number of outer iterations')
    parser.add_argument('--local_iter', type=int, default=100, metavar='LIT', help='number of local iterations')
    parser.add_argument('--match_iter', type=int, default=10, metavar='MIT', help='number of model matching iterations')
    
    parser.add_argument('--root', type=str, default='./multiview', metavar='N', help='where is data set')
    parser.add_argument('--views_dir', type=str, default='./views.txt', metavar='N', help='where is view partition file')
    parser.add_argument('--classes_dir', type=str, default='./classes.txt', metavar='N', help='where is class partition file')
    parser.add_argument('--decay', type=str, default='./decay.txt', metavar='DEC', help='learning rate decay schedule file')
    parser.add_argument('--log_dir', type=str, default='./log', metavar='N', help='where is log file')
    parser.add_argument('--tag', type=str, default='DE', metavar='T', help='task name')
    args = parser.parse_args()
    args.logfile = args.log_dir+'/'+args.tag+'_M'+args.model+'_A'+str(args.num_agents)+'_B'+str(args.batch_size)+'_C'+str(args.comm_rounds)+'_V'+str(args.total_views)+'_LR'+str(args.lr)+'.csv'

    with open(args.logfile,'w') as fp:
        print('model, agent, batchsize, lr, comm_iter, local_iter, matching_iter',file= fp)
        print(args.model, args.num_agents, args.batch_size, args.lr, args.comm_rounds, args.local_iter, args.match_iter, sep=',', file = fp)
        print('iter, local_acc, local_loss, local_acc_post, local_loss_post, global_acc, global_loss', file=fp)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    if args.model.startswith('resnet18'):
        args.layer_num = 3
    elif args.model.startswith('resnet'):
        args.layer_num = 2
    else:
        args.layer_num = 4
    return args

def save_log(global_iter, local_acc, local_loss, local_acc_post, local_loss_post, global_acc, global_loss, global_shape, args):
    l_acc = sum(local_acc)/len(local_acc)
    l_loss = sum(local_loss)/len(local_loss)
    l_acc_p = sum(local_acc_post)/len(local_acc_post)
    l_loss_p = sum(local_loss_post)/len(local_loss_post)
    with open(args.logfile,'a') as fp:
        print(global_iter,l_acc,l_loss.item(),l_acc_p,l_loss_p.item(), global_acc, global_loss.item(),sep = ',', file=fp, flush = True)
    if(global_iter%10 !=0):
        print(global_shape)