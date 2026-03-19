import os
import cv2
import lmdb
import torch
import jpegio
import tempfile
import numpy as np
import pickle
import six
import warnings
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

        # 在线数据增强：当前蒸馏阶段统一使用干净输入（关闭噪声/模糊/强JPEG增强）
        self.clean_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.455, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        # 针对偶发坏样本（JPEG 结构异常等）做容错重试，避免 DataLoader worker 直接崩溃
        for retry in range(3):
            cur_index = (index + retry) % self.max_nums
            try:
                with self.envs.begin(write=False) as txn:
                    img_key = 'image-%09d' % cur_index
                    imgbuf = txn.get(img_key.encode('utf-8'))
                    lbl_key = 'label-%09d' % cur_index
                    lblbuf = txn.get(lbl_key.encode('utf-8'))
                    if imgbuf is None or lblbuf is None:
                        raise RuntimeError(f'Missing lmdb entry at index={cur_index}')

                    buf = six.BytesIO()
                    buf.write(imgbuf)
                    buf.seek(0)
                    im = Image.open(buf).convert('RGB')
                    mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8)
                    record = self.record[cur_index]
                    choicei = len(record) - 1
                    q = int(record[-1])
                    use_qtb = self.pks[q]
                    q_seq = []
                    if choicei > 1:
                        q2 = int(record[-3])
                        q_seq.append(q2)
                    if choicei > 0:
                        q1 = int(record[-2])
                        q_seq.append(q1)
                    q_seq.append(q)

                    # 统一调整 mask 到 512x512，并转为张量
                    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                    mask_tensor = self.totsr(image=mask_resized.copy())['image']

                    def _extract_dct_with_quality_chain(pil_rgb):
                        with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
                            pil_gray = pil_rgb.convert("L")
                            for qv in q_seq:
                                pil_gray.save(tmp.name, "JPEG", quality=qv)
                                pil_gray = Image.open(tmp.name).copy()
                            try:
                                jpg = jpegio.read(tmp.name)
                                dct = jpg.coef_arrays[0].copy()
                            except Exception as exc:
                                warnings.warn(
                                    f'jpegio read failed at index={cur_index}, fallback zero DCT: {exc}',
                                    RuntimeWarning
                                )
                                # fallback 维持可训练，避免 worker 崩溃
                                dct = np.zeros((512, 512), dtype=np.float32)
                            return dct, pil_gray.convert('RGB')

                    # ---------- 干净图像 ----------
                    dct_clean, im_clean_rgb = _extract_dct_with_quality_chain(im)

                    # ---------- 使用 Albumentations 生成干净图 ----------
                    im_clean_np = np.array(im_clean_rgb)
                    clean_aug = self.clean_transform(image=im_clean_np)
                    img_clean = clean_aug['image']
                    # 学生输入与教师输入保持一致：同一份干净图像与干净 DCT
                    img_dist = img_clean.clone()
                    dct_dist = dct_clean.copy()

                    return {
                        'img_clean': img_clean,
                        'img_dist': img_dist,
                        'dct_clean': np.clip(np.abs(dct_clean), 0, 20),
                        'dct_dist': np.clip(np.abs(dct_dist), 0, 20),
                        'qtb': use_qtb,
                        'mask': mask_tensor.long(),
                    }
            except Exception as exc:
                if retry == 2:
                    raise
                warnings.warn(
                    f'Bad sample at index={cur_index}, retry with next sample: {exc}',
                    RuntimeWarning
                )
