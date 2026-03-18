import os
import time
import cv2
import lmdb
import torch
import jpegio
import numpy as np
import torch.nn as nn
import math
import logging
import pickle
import six
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile
from functools import partial
import torch.nn.functional as F

from .losses import SoftCrossEntropyLoss, LovaszLoss
from .light_dtd import LightDTD


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./')  # root to the dir of lmdb files
parser.add_argument('--pth', type=str, default='light_dtd.pth')
parser.add_argument('--lmdb_name', type=str, default='DocTamperV1-FCD')
parser.add_argument('--minq', type=int, default=75)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


class TamperDataset(Dataset):
    def __init__(self, roots, mode, minq=95, qtb=90, max_readers=64, dataset_name=None):
        self.envs = lmdb.open(roots, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        self.max_nums = self.nSamples
        self.minq = minq
        self.mode = mode
        with open('qt_table.pk', 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k, v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        name_for_pks = dataset_name if dataset_name is not None else os.path.basename(os.path.normpath(roots))
        pks_path = os.path.join('pks', '%s_%d.pk' % (name_for_pks, minq))
        with open(pks_path, 'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))]
        )

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)
            H, W = mask.shape
            record = self.record[index]
            choicei = len(record) - 1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei > 1:
                q2 = int(record[-3])
                use_qtb2 = self.pks[q2]
            if choicei > 0:
                q1 = int(record[-2])
                use_qtb1 = self.pks[q1]
            mask = self.totsr(image=mask.copy())['image']
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                im = im.convert("L")
                if choicei > 1:
                    im.save(tmp, "JPEG", quality=q2)
                    im = Image.open(tmp)
                if choicei > 0:
                    im.save(tmp, "JPEG", quality=q1)
                    im = Image.open(tmp)
                im.save(tmp, "JPEG", quality=q)
                jpg = jpegio.read(tmp.name)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct), 0, 20),
                'q': use_qtb,
                'i': q
            }


lmdb_path = os.path.join(args.data_root, args.lmdb_name)
test_data = TamperDataset(lmdb_path, False, minq=args.minq, dataset_name=args.lmdb_name)


class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)
        )
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def eval_net_light_dtd(model, test_data, plot=False, device='cuda'):
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    LovaszLoss_fn = LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    ckpt = torch.load(args.pth, map_location='cpu')
    state_dict = ckpt.get('student_state', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    iou = IOUMetric(2)
    precisons = []
    recalls = []

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(test_loader)):
            data, target, dct_coef, qs, q = (
                batch_samples['image'],
                batch_samples['label'],
                batch_samples['rgb'],
                batch_samples['q'],
                batch_samples['i'],
            )
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            dct_coef = Variable(dct_coef.to(device))
            qs = Variable(qs.unsqueeze(1).to(device))

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            pred = model(data, dct_coef, qs)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start_time

            bs = data.size(0)
            total_time += elapsed
            total_samples += bs

            predt = pred.argmax(1)
            pred_np = pred.cpu().data.numpy()
            targt = target.squeeze(1)
            matched = (predt * targt).sum((1, 2))
            pred_sum = predt.sum((1, 2))
            target_sum = targt.sum((1, 2))
            precisons.append((matched / (pred_sum + 1e-8)).mean().item())
            recalls.append((matched / target_sum).mean().item())
            pred_cls = np.argmax(pred_np, axis=1)
            iou.add_batch(pred_cls, target.cpu().data.numpy())

    acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
    precisons = np.array(precisons).mean()
    recalls = np.array(recalls).mean()
    f1 = (2 * precisons * recalls / (precisons + recalls + 1e-8))

    fps = total_samples / total_time if total_time > 0 else 0.0
    print('[val] iou:{} pre:{} rec:{} f1:{} fps:{:.2f}'.format(iu, precisons, recalls, f1, fps))


def main():
    device = args.device
    model = LightDTD(fph_out_channels=128, pretrained_vph=False, classes=2)
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    eval_net_light_dtd(model, test_data, device=device)


if __name__ == '__main__':
    main()

