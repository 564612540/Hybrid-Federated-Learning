#!/bin/bash -l        
#PBS -l walltime=24:00:00,nodes=1:ppn=24:gpus=1,mem=24gb 
#PBS -m abe 
#PBS -M zhan6234@umn.edu 
cd ./HybridFL
conda activate /panfs/roc/groups/5/mhong/zhan6234/my_ws
module load cuda cuda-sdk

python3.8 ./train_hybrid.py --tag F4_D1 --num_agents 4 --batch_size 32 --comm_rounds 32 --local_iter 32 --match_iter 4 --root ./data/modelnet40v2 --model resnet34 --lr 0.005 --total_views 4 --classes_dir ./classes.txt --views_dir ./views.txt
python3.8 ./train_hybrid_1.py --tag F4_P1 --num_agents 4 --batch_size 32 --comm_rounds 32 --local_iter 32 --match_iter 4 --root ./data/modelnet40v2 --model resnet34 --lr 0.005 --total_views 4 --classes_dir ./classes.txt --views_dir ./views.txt
