# HybridFL

Code for paper Hybrid Federated Learning: Algorithms and Implementation https://arxiv.org/abs/2012.12420

Usage: run the following script:

python3 ./train_hybrid.py --num_agents 3 --batch_size 16 --comm_rounds 32 --local_iter 24 --match_iter 4 --root ./data/modelnet40v1 --model resnet18 --lr 0.005

