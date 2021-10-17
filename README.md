# HybridFL

Code for paper Hybrid Federated Learning: Algorithms and Implementation https://arxiv.org/abs/2012.12420

Usage: 

Dataset: http://maxwell.cs.umass.edu/mvcnn-data/

Run the following script to train the models:

python3 ./train_hybrid.py --tag F4_D1 --num_agents 4 --batch_size 32 --comm_rounds 32 --local_iter 32 --match_iter 4 --root ./data/modelnet40v2 --model resnet34 --lr 0.005 --total_views 4 --classes_dir ./classes.txt --views_dir ./views.txt

