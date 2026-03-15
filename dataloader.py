import os
import cv2
import lmdb
import torch
import jpegio
import tempfile
import numpy as np
import pickle
import six
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision

class DocTamperDataset(Dataset):
    '''
    A basic dataloader of the inference mode

    roots: path of the LMDB file
    minq : min random compression factor, choiced from (75, 80, 85, 90)
    max_readers : max_readers of the LMDB loader
    '''
    def __init__(self, roots, minq=75, max_nums = None, max_readers=64):
        self.envs = lmdb.open(roots,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False)
        with self.envs.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        if max_nums is None:
            self.max_nums = self.nSamples
        else:
            self.max_nums = min(max_nums, self.nSamples)
        self.minq = minq # Q
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lmdb_name = os.path.basename(os.path.normpath(roots))
        qt_table_path = os.path.join(base_dir, 'qt_table.pk')
        pks_record_path = os.path.join(base_dir, 'pks', f'{lmdb_name}_{minq}.pk')

        with open(qt_table_path, 'rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        with open(pks_record_path, 'rb') as f: # random compression factors with the same random seed
            self.record = pickle.load(f)
        self.totsr = ToTensorV2()
        # 基础归一化与张量化（与原始逻辑保持一致）
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.455, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        # 在线数据增强：干净图与失真图的 Albumentations 管道
        self.clean_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.455, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.distort_transform = A.Compose([
            A.Resize(512, 512),
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0),
            A.Normalize(mean=(0.485, 0.455, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

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
            mask = (cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)!=0).astype(np.uint8)
            record = self.record[index]
            choicei = len(record)-1
            q = int(record[-1])
            if True:
                use_qtb = self.pks[q]
                if choicei>1:
                    q2 = int(record[-3])
                    use_qtb2 = self.pks[q2]
                if choicei>0:
                    q1 = int(record[-2])
                    use_qtb1 = self.pks[q1]

            # 统一调整 mask 到 512x512，并转为张量
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_tensor = self.totsr(image=mask_resized.copy())['image']

            # ---------- 干净图像的 DCT 提取（保持原逻辑） ----------
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                im_gray = im.convert("L")
                if True:
                    if choicei>1:
                        im_gray.save(tmp, "JPEG", quality=q2)
                        im_gray = Image.open(tmp)
                    if choicei>0:
                        im_gray.save(tmp, "JPEG", quality=q1)
                        im_gray = Image.open(tmp)
                    im_gray.save(tmp, "JPEG", quality=q)
                jpg_clean = jpegio.read(tmp.name)
                dct_clean = jpg_clean.coef_arrays[0].copy()
                im_clean_rgb = im_gray.convert('RGB')

            # ---------- 使用 Albumentations 生成干净 / 失真图 ----------
            im_clean_np = np.array(im_clean_rgb)
            clean_aug = self.clean_transform(image=im_clean_np)
            img_clean = clean_aug['image']

            # 以干净图为基准进行失真增强
            dist_aug = self.distort_transform(image=im_clean_np)
            img_dist = dist_aug['image']
            im_dist_np = dist_aug['image'].permute(1, 2, 0).cpu().numpy()
            im_dist_np = (im_dist_np * np.array([0.229, 0.224, 0.225])[None, None, :] +
                          np.array([0.485, 0.455, 0.406])[None, None, :])
            im_dist_np = np.clip(im_dist_np * 255.0, 0, 255).astype(np.uint8)
            im_dist_rgb = Image.fromarray(im_dist_np)

            # ---------- 失真图像的 DCT 提取（沿用相同 JPEG+DCT 逻辑） ----------
            with tempfile.NamedTemporaryFile(delete=True) as tmp2:
                im_dist_gray = im_dist_rgb.convert("L")
                if True:
                    if choicei>1:
                        im_dist_gray.save(tmp2, "JPEG", quality=q2)
                        im_dist_gray = Image.open(tmp2)
                    if choicei>0:
                        im_dist_gray.save(tmp2, "JPEG", quality=q1)
                        im_dist_gray = Image.open(tmp2)
                    im_dist_gray.save(tmp2, "JPEG", quality=q)
                jpg_dist = jpegio.read(tmp2.name)
                dct_dist = jpg_dist.coef_arrays[0].copy()

            return {
                'img_clean': img_clean,
                'img_dist': img_dist,
                'dct_clean': np.clip(np.abs(dct_clean), 0, 20),
                'dct_dist': np.clip(np.abs(dct_dist), 0, 20),
                'qtb': use_qtb,
                'mask': mask_tensor.long(),
            }
