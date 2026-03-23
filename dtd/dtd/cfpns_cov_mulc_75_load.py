import os
import cv2
import lmdb
import torch
import jpegio
import numpy as np
import torch.nn as nn
import gc
import math
import time
import copy
import logging
import torch.optim as optim
import torch.distributed as dist
import random
import pickle
import six
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
import albumentations as A
from dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./')
parser.add_argument('--train_name', type=str, default='CHDOC_JPEG0')
parser.add_argument('--model_name', type=str, default='catneto')
parser.add_argument('--att', type=str, default='None')
parser.add_argument('--num', type=str, default='1')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--bs', type=int, default=6)
parser.add_argument('--es', type=int, default=0)
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--xk', type=int, default=0)
parser.add_argument('--numw', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--pilt', type=int, default=0)
parser.add_argument('--base', type=int, default=1)
parser.add_argument('--lr_base', type=float, default=3e-4)
parser.add_argument('--cp', type=float, default=1.0)
parser.add_argument('--mode', type=str, default='0123')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--adds', type=str, default='123')
parser.add_argument('--lossw', type=str, default='1,2,3,4')
import random
class CMDataset(Dataset):
    def __init__(self, roots, mode, i0, steps=8192, pilt=False, casia=False,ranger=1, max_nums = None, max_readers=64):
        self.dict = {}
        self.cnts = []
        self.lens = []
        self.envs = []
        self.more_than_one = (len(roots)>1)
        self.pilt = pilt
        self.casia = casia
        self.ranger = ranger
        for root in roots:
            if '$' in root:
                root_use, nums = root.split('$')
                nums = int(nums)
                self.envs.append(lmdb.open(root_use,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False))
                with self.envs[-1].begin(write=False) as txn:
                    str = 'num-samples'.encode('utf-8')
                    nSamples = int(txn.get(str))
                    if not (max_nums is None):
                        nSamples = min(nSamples,max_nums)
                self.lens.append(nSamples*nums)
                self.cnts.append(nSamples)
            else:
                self.envs.append(lmdb.open(root,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False))
                with self.envs[-1].begin(write=False) as txn:
                    str = 'num-samples'.encode('utf-8')
                    nSamples = int(txn.get(str))
                    if not (max_nums is None):
                        nSamples = min(nSamples,max_nums)
                self.lens.append(nSamples)
                self.cnts.append(nSamples)
        self.lens = np.array(self.lens)
        self.sums = np.cumsum(self.lens)
        self.len_sum = len(self.sums)
        self.nSamples = self.lens.sum()
        self.i0 = i0
        self.steps = steps
        print('*'*60)
        print('Dataset inited!',i0,pilt)
        print('*'*60)
        self.mode = mode
        with open('qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.Tensor(v)
        npr = np.arange(self.nSamples)
        if True:#shuffle:
            np.random.seed(i0)
            self.idxs = np.random.choice(self.nSamples,self.nSamples,replace=False)
        else:
            self.idxs = npr
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def calnum(self,num):
        if num<self.lens[0]:
            return 0,num%(self.cnts[0])
        else:
            for li,l in enumerate(self.sums):
                if ((l<=num) and ((li==self.len_sum) or (num<self.sums[li+1]))):
                    return (li+1),((num-l)%(self.cnts[li+1]))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        itm_num = self.idxs[idx]
        env_num,index = self.calnum(itm_num)
        with self.envs[env_num].begin(write=False) as txn:
            if True:
                img_key = 'image-%09d' % index
                imgbuf = txn.get(img_key.encode('utf-8'))
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                im = Image.open(buf)
                lbl_key = 'label-%09d' % index
                lblbuf = txn.get(lbl_key.encode('utf-8'))
                info_key = 'info-%09d' % index
                infobuf = txn.get(info_key.encode('utf-8'))
                mask = (cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)!=0).astype(np.uint8)
                H,W = mask.shape
                if ((H!=512) or (W!=512)):
                    return self.__getitem__(random.randint(0,self.nSamples-1))
                if ((idx+self.i0)<600000):
                    q = random.randint(100-np.clip((idx+self.i0)*random.uniform(0,1)//self.steps,0,25),100) # random.randint(75,100)
                    q2 = random.randint(100-np.clip((idx+self.i0)*random.uniform(0,1)//self.steps,0,25),100)
                    q3 = random.randint(100-np.clip((idx+self.i0)*random.uniform(0,1)//self.steps,0,25),100)
                else:
                    q = random.randint(75,100)
                    q2 = random.randint(75,100)
                    q3 = random.randint(75,100)
                use_qtb = self.pks[q] # random.choice(self.qt75[q])
                if random.uniform(0,1) < 0.5:
                    im = im.rotate(90)
                    mask = np.rot90(mask,1)
                mask = self.totsr(image=mask.copy())['image']
                if random.uniform(0,1) < 0.5:
                    im = self.hflip(im)
                    mask = self.hflip(mask)
                if random.uniform(0,1) < 0.5:
                    im = self.vflip(im)
                    mask = self.vflip(mask)
                with tempfile.NamedTemporaryFile(delete=True,prefix=str(idx)) as tmp:
                    im = im.convert("L")
                    if True:
                        if '1' in infobuf:
                            choicei=0
                        else:
                            choicei = random.randint(0,2)
                        if choicei>1:
                            im.save(tmp,"JPEG",quality=q3)
                            im=Image.open(tmp)
                        if choicei>0:
                            im.save(tmp,"JPEG",quality=q2)
                            im=Image.open(tmp)
                        im.save(tmp,"JPEG",quality=q)
                    jpg = jpegio.read(tmp.name)
                    dct = jpg.coef_arrays[0].copy()
                    im = im.convert('RGB')
                return {
                    'image': self.toctsr(im),
                    'label': mask.long(),
                    'rgb': np.clip(np.abs(dct),0,20),
                    'q':use_qtb,
                    'i':q
                }
            else:
                print('data error')
                return self.__getitem__(random.randint(0,self.nSamples-1))

args = parser.parse_args()
device = torch.device(args.local_rank)
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl')
use_pilt = (args.pilt==1)
ngpu = torch.cuda.device_count()
ngpub = ngpu * args.base
if ngpu > 1:
    gpus = True
else:
    gpus = False

each_step = (args.xk==0)
train_path = args.data_root+args.train_name
trainps = [args.model_name]
train_data1 = CMDataset([(args.data_root+ps) for ps in trainps],False,0,pilt=use_pilt)
train_data2 = None#CMDataset([(args.data_root+ps) for ps in trainps2],False,0)#RSCDataset(args.data_root+'SN1_DF',True,0)
train_data3 = None
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from mmseg.utils import get_root_logger

model=seg_dtd(args.model_name,args.n_class).to(device)
if gpus:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model= torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
model_name = args.model_name
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name)
try:
  if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
except:
  pass
try:
  if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
except:
  pass
import gc
# 参数设置
param = {}
param['batch_size'] = args.bs       # 批大小
param['epochs'] = args.ep       # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 64     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=int(24/ngpub)  #cosine warmup的参数
param['load_ckpt_dir'] = None

def train_net_qyl(param, model, train_data1, train_data2, train_data3, plot=False,device='cuda'):
    # 初始化参数
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    iter_inter      = param['iter_inter']
    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']
    T0=param['T0']
    scaler = GradScaler()
    lr_base = args.lr_base 
    train_data_size = train_data1.__len__()#+train_data2.__len__()
    train_loader1 = iter(DataLoader(dataset=train_data1, batch_size=batch_size, num_workers=args.numw, shuffle=False))
    optimizer = optim.AdamW(model.parameters(), lr=3e-4 ,weight_decay=5e-4)
    iter_per_epoch = len(train_loader1)
    totalstep = epochs*iter_per_epoch
    warmupr = 1/epochs
    warmstep = 200
    lr_min = 1e-5
    lr_min /= lr_base
    lr_dict = {i:((((1+math.cos((i-warmstep)*math.pi/(totalstep-warmstep)))/2)+lr_min) if (i > warmstep) else (i/warmstep+lr_min)) for i in range(totalstep)}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_dict[epoch])
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    logger = get_logger(os.path.join(save_log_dir, time.strftime("%m-%d", time.localtime()) +'_'+model_name+ '.log'))
    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_iou = 0
    best_epoch=0
    epoch_start = 0
    lossw = [float(lw) for lw in args.lossw.split(',')]
    lossws = sum(lossw)
    if ((args.load!=0) and (args.es!=0)):
        ckpt = torch.load(os.path.join(save_ckpt_dir, 'checkpoint-best.pth'),map_location='cpu')
        epoch_start = (ckpt['epoch']+1)
        assert epoch_start==args.es,'{}!={}'.format(epoch_start,args.es)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    logger.info('Total Epoch:{} Training num:{}  Validation num:{}'.format(epochs, train_data_size, 0))
    for epoch in range(epoch_start, epochs):
        tmp_i = epoch*train_data_size
        iter_i = epoch*iter_per_epoch
        if (epoch!=0):
            train_data1 = CMDataset([(args.data_root+ps) for ps in trainps],False,tmp_i,pilt=use_pilt)
            train_loader1 = iter(DataLoader(dataset=train_data1, batch_size=batch_size, num_workers=args.numw, shuffle=False))
        epoch_start = time.time()
        # 训练阶段
        train_nums = [0]*len(train_loader1)#+[1]*len(train_loader2)#+[2]*len(train_loader3)
        random.shuffle(train_nums)
        train_loader_size = len_train = len(train_nums)
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx in range(len(train_nums)):#len_train):
            this_train_id = train_nums[batch_idx]
            if this_train_id==0:
                if True:
                    batch_samples = next(train_loader1)
                else:
                    print('error')
                    continue
            data, target, catnetinput, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'],batch_samples['i']
            data, target, catnetinput, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(catnetinput.to(device)), Variable(qs.unsqueeze(1).to(device))
            with autocast(): #need pytorch>1.6
                pred = model(data,catnetinput,qs)
                loss = 1.*LovaszLoss_fn(pred, target)+SoftCrossEntropy_fn(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler.step(iter_i+batch_idx) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} {} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg,this_train_id,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                print('qs',q)
                train_iter_loss.reset()

        with open('tmp_record.pk','wb') as f:
            pickle.dump(epoch+1,f)
        if True:#iu[1] > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
            
    return 

train_net_qyl(param, model, train_data1, train_data2, train_data3, device=device)


