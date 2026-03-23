import os
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--py', type=str, default='cfpns_cov_mulc_75_load.py')
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--numw', type=int, default=16)
parser.add_argument('--model_name', type=str, default='DocTamperV1-TrainingSet')
args = parser.parse_args()
error_cnt=0
if True:
    if os.path.exists('tmp_record.pk'):
        with open('tmp_record.pk','rb') as f:
            eps = pickle.load(f)
            if eps==args.ep:
                print('work done')
                exit(0)
    else:
        eps = 0
    if eps!=0:
        error_cnt = (error_cnt+1)
        if error_cnt==2:
            numw = args.numw//2
        elif error_cnt==3:
            numw = 0
        else:
            numw = args.numw
        print('$'*60)
        print('restart')
        print('$'*60)
        os.system('CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 {} --model_name {} --ep {} --load 1 --es {} --numw {}'.format(args.py,args.model_name,args.ep,eps,numw))
    else:
        os.system('CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 {} --model_name {} --ep {} --load 0 --es 0 --numw {}'.format(args.py,args.model_name,args.ep,args.numw))
