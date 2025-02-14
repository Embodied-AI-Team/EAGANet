import os
import sys
from argparse import ArgumentParser

from graspdatagen import GraspDataGen


'''
Run Command : 

python gen_offline_data.py \
  --data_dir ../data/grasp_train_data \
  --data_fn ../grasp_stats/train_data_list.txt \
  --num_processes 6 \
  --num_epochs 150 \
  

python -m pdb gen_offline_data.py --data_dir ../grasp_train_data --data_fn grasp_stats/train_data_list.txt --num_processes 6 --num_epochs 150 
'''
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--data_fn', type=str, help='data file that indexs all shape-ids')
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
parser.add_argument('--num_epochs', type=int, default=1000, help='control the data amount')
parser.add_argument('--starting_epoch', type=int, default=0, help='if you want to run this data generation across multiple machines, you can set this parameter so that multiple data folders generated on different machines have continuous trial-id for easier merging of multiple datasets')
parser.add_argument('--out_fn', type=str, default=None, help='a file that lists all valid interaction data collection [default: None, meaning data_tuple_list.txt]. Again, this is used when you want to generate data across multiple machines. You can store the filelist on different files and merge them together to get one data_tuple_list.txt')
conf = parser.parse_args()

if conf.out_fn is None:
    conf.out_fn = 'data_tuple_list.txt'

datagen = GraspDataGen(conf.num_processes)

with open(conf.data_fn, 'r') as fin: # open file on only read mode, if file not exist, is not created.
    for l in fin.readlines():
        cat = l.rstrip().split()[0]
        for epoch in range(conf.starting_epoch, conf.starting_epoch+conf.num_epochs):
            print(cat, epoch)
            datagen.add_one_collect_job(conf.data_dir, cat, epoch)

datagen.start_all() # 对每个Job调用 collect_data.py 进行处理

data_tuple_list = datagen.join_all()
with open(os.path.join(conf.data_dir, conf.out_fn), 'w') as fout:
    for item in data_tuple_list:
        fout.write(item.split('/')[-1]+'\n')

