import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime

import cv2
import lmdb
import numpy as np
import pickle
import six
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from albumentations.pytorch import ToTensorV2
import torchvision
import jpegio

from dataloader import DocTamperDataset
from models.dtd import seg_dtd
from models.light_dtd import LightDTD
from models.losses import SoftBCEWithLogitsLoss, LovaszLoss, DiceLoss
from models.losses.distill_loss import DistillationLoss


# ---------------------------------------------------------------------------
# Eval dataset & metrics (self-contained, no module-level argparse)
# ---------------------------------------------------------------------------

class TamperDataset(Dataset):
    """eval_light_dtd.py 中的 TamperDataset，移至此处避免 argparse 冲突。"""

    def __init__(self, roots, minq=75, max_readers=64, dataset_name=None,
                 base_dir=None):
        self._roots = roots
        self._max_readers = max_readers
        self.envs = None  # lazy open per worker
        self.minq = minq

        env = lmdb.open(roots, max_readers=max_readers, readonly=True,
                        lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        env.close()
        self.max_nums = self.nSamples

        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        qt_path = os.path.join(base_dir, 'qt_table.pk')
        with open(qt_path, 'rb') as f:
            pks = pickle.load(f)
        self.pks = {k: torch.LongTensor(v) for k, v in pks.items()}

        name = dataset_name or os.path.basename(os.path.normpath(roots))
        pks_path = os.path.join(base_dir, 'pks', f'{name}_{minq}.pk')
        with open(pks_path, 'rb') as f:
            self.record = pickle.load(f)

        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _ensure_env(self):
        if self.envs is None:
            self.envs = lmdb.open(
                self._roots, max_readers=self._max_readers, readonly=True,
                lock=False, readahead=False, meminit=False)

    def close(self):
        if self.envs is not None:
            self.envs.close()
            self.envs = None

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        self._ensure_env()
        with self.envs.begin(write=False) as txn:
            imgbuf = txn.get(f'image-{index:09d}'.encode())
            lblbuf = txn.get(f'label-{index:09d}'.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
                    != 0).astype(np.uint8)
            record = self.record[index]
            choicei = len(record) - 1
            q = int(record[-1])
            use_qtb = self.pks[q]
            mask = self.totsr(image=mask.copy())['image']
            with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
                im = im.convert("L")
                if choicei > 1:
                    im.save(tmp.name, "JPEG", quality=int(record[-3]))
                    im = Image.open(tmp.name).copy()
                if choicei > 0:
                    im.save(tmp.name, "JPEG", quality=int(record[-2]))
                    im = Image.open(tmp.name).copy()
                im.save(tmp.name, "JPEG", quality=q)
                jpg = jpegio.read(tmp.name)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct), 0, 20),
                'q': use_qtb,
            }


class IOUMetric:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        return np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        iu = np.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        return acc, iu, mean_iu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_prefix(state_dict, prefix):
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()}


def _load_teacher_state_dict(teacher, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(raw)}")
    model_keys = set(teacher.state_dict().keys())
    candidates = [
        raw,
        _strip_prefix(raw, 'module.'),
        _strip_prefix(raw, 'model.'),
        _strip_prefix(_strip_prefix(raw, 'module.'), 'model.'),
    ]
    best = max(candidates, key=lambda s: len(model_keys & set(s.keys())))
    teacher.load_state_dict(best, strict=True)


def _load_student_state_dict(student, raw):
    model_keys = set(student.state_dict().keys())
    candidates = [
        raw,
        _strip_prefix(raw, 'module.'),
        _strip_prefix(raw, 'model.'),
        _strip_prefix(_strip_prefix(raw, 'module.'), 'model.'),
    ]
    best = max(candidates, key=lambda s: len(model_keys & set(s.keys())))
    student.load_state_dict(best, strict=True)


def _patch_legacy_gelu(module):
    for m in module.modules():
        if isinstance(m, nn.GELU) and not hasattr(m, "approximate"):
            m.approximate = "none"
        if m.__class__.__name__ == "DropPath" and not hasattr(m, "scale_by_keep"):
            m.scale_by_keep = True


def _format_teacher_qtable(batch, dct_tensor, device):
    qtb = batch.get('qtb')
    if qtb is None:
        qtb = torch.zeros(dct_tensor.size(0), 8, 8, dtype=torch.long,
                          device=device)
    else:
        qtb = qtb.to(device)
    if qtb.dim() == 2:
        qtb = qtb.view(qtb.size(0), 8, 8)
    if qtb.dim() == 3:
        qtb = qtb.unsqueeze(1)
    return qtb.long()


def _student_state(student):
    if isinstance(student, nn.DataParallel):
        return student.module.state_dict()
    return student.state_dict()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{ts}.log')
    metrics_file = os.path.join(log_dir, f'metrics_{ts}.jsonl')

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, metrics_file


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(student, dataset_name, lmdb_path, minq, device, batch_size=6,
             num_workers=2, base_dir=None):
    import gc
    test_ds = TamperDataset(lmdb_path, minq=minq, dataset_name=dataset_name,
                            base_dir=base_dir)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False,
                        persistent_workers=False)
    student.eval()
    iou_metric = IOUMetric(2)
    precisions, recalls = [], []

    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)
        dct = batch['rgb'].to(device, non_blocking=True)
        qs = batch['q'].to(device, non_blocking=True).view(img.size(0), -1).long()

        pred = student(img, dct, qs)
        pred_cls = pred.argmax(1)
        targt = target.squeeze(1)
        matched = (pred_cls * targt).sum((1, 2))
        precisions.append((matched / (pred_cls.sum((1, 2)) + 1e-8)).mean().item())
        recalls.append((matched / (targt.sum((1, 2)) + 1e-8)).mean().item())
        iou_metric.add_batch(pred_cls.cpu().numpy(), target.cpu().numpy())
        del img, target, dct, qs, pred, pred_cls, targt, matched

    acc, iu, mean_iu = iou_metric.evaluate()
    prec = np.mean(precisions)
    rec = np.mean(recalls)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    del loader
    test_ds.close()
    del test_ds
    gc.collect()
    torch.cuda.empty_cache()

    student.train()
    return {
        'dataset': dataset_name,
        'acc': float(acc),
        'iou_bg': float(iu[0]),
        'iou_tamper': float(iu[1]),
        'mean_iou': float(mean_iu),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./')
    p.add_argument('--lmdb_name', type=str, default='DocTamperV1-TrainingSet')
    p.add_argument('--minq', type=int, default=75)
    p.add_argument('--teacher_pth', type=str, default='pths/dtd_doctamper.pth')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=96)
    p.add_argument('--eval_batch_size', type=int, default=6)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eta_min', type=float, default=0.0,
                   help='CosineAnnealingLR 最低 lr，0 表示不使用 scheduler')
    p.add_argument('--weight_decay', type=float, default=0.0,
                   help='AdamW weight decay，>0 时使用 AdamW 替代 Adam')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=0.01)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='pths')
    p.add_argument('--save_interval', type=int, default=5)
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--resume', type=str, default='',
                   help='checkpoint 路径，留空则自动检测 latest.pth')
    p.add_argument('--reset_optimizer', action='store_true',
                   help='加载权重但重置 optimizer/scheduler/epoch（用于二阶段微调）')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_models(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    teacher = seg_dtd('', 2).to(device)
    _patch_legacy_gelu(teacher)
    _load_teacher_state_dict(teacher, args.teacher_pth)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = LightDTD(fph_out_channels=64, pretrained_vph=True,
                       classes=2).to(device)
    _patch_legacy_gelu(student)
    student.train()

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    return teacher, student, device


def build_criterion(args):
    bce = SoftBCEWithLogitsLoss()
    lovasz = LovaszLoss(mode='multiclass')
    dice = DiceLoss(mode='multiclass', from_logits=True, smooth=1.0)

    class HardLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce, self.lovasz, self.dice = bce, lovasz, dice

        def forward(self, logits, target):
            if target.dim() == 3:
                target_ = target.unsqueeze(1).float()
            else:
                target_ = target.float()
            target_long = target.squeeze(1).long()
            return (0.5 * self.bce(logits[:, 1:2], target_)
                    + self.lovasz(logits, target_long)
                    + self.dice(logits, target_long))

    return DistillationLoss(
        hard_loss_fn=HardLoss(),
        temperature=2.0,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
    )


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(epoch, teacher, student, dataloader, criterion, optimizer,
                    scaler, device):
    teacher.eval()
    student.train()

    feat_cache = {'teacher': None, 'student': None}

    teacher_fu = (teacher.module.model.FU if isinstance(teacher, nn.DataParallel)
                  else teacher.model.FU)
    student_adapt = (student.module.adapt_layer if isinstance(student, nn.DataParallel)
                     else student.adapt_layer)

    def _hook_t(_m, _i, o):
        feat_cache['teacher'] = o.detach()

    def _hook_s(_m, _i, o):
        feat_cache['student'] = o

    h_t = teacher_fu.register_forward_hook(_hook_t)
    h_s = student_adapt.register_forward_hook(_hook_s)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', ncols=120)
    sums = {'loss': 0., 'hard': 0., 'soft': 0., 'feat': 0.}
    n = 0

    for batch in pbar:
        img = batch['image'].to(device, non_blocking=True)
        dct = batch['dct'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        with torch.no_grad():
            qt_t = _format_teacher_qtable(batch, dct, device)
            t_logits = teacher(img, dct, qt_t)
            t_feat = feat_cache['teacher']

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            qt_s = (batch['qtb'].to(device, non_blocking=True)
                    .view(dct.size(0), -1).long()
                    if 'qtb' in batch
                    else dct.new_zeros(dct.size(0), 64, dtype=torch.long))
            s_logits = student(img, dct, qt_s)
            s_feat = feat_cache['student']
            loss, l_hard, l_soft, l_feat = criterion(
                s_logits, t_logits, s_feat, t_feat, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n += 1
        sums['loss'] += loss.item()
        sums['hard'] += l_hard.item()
        sums['soft'] += l_soft.item()
        sums['feat'] += l_feat.item()
        pbar.set_postfix({k: f'{v / n:.4f}' for k, v in sums.items()})

    h_t.remove()
    h_s.remove()
    return {k: v / max(1, n) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, epoch, student, optimizer, scaler, best_iou,
                    scheduler=None):
    state = {
        'epoch': epoch,
        'student_state': _student_state(student),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'best_iou': best_iou,
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, student, optimizer, scaler, scheduler=None,
                    reset_optimizer=False):
    ckpt = torch.load(path, map_location='cpu')
    raw = ckpt.get('student_state', ckpt.get('state_dict'))
    if raw is None:
        raise KeyError(f"Checkpoint lacks student_state/state_dict: {path}")
    _load_student_state_dict(student, raw)

    if reset_optimizer:
        return 0, float(ckpt.get('best_iou', 0.0))

    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return int(ckpt.get('epoch', 0)), float(ckpt.get('best_iou', 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger, metrics_file = setup_logging(args.log_dir)
    base_dir = os.path.abspath(args.data_root)

    logger.info(f'Args: {vars(args)}')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data ----
    lmdb_path = os.path.join(args.data_root, args.lmdb_name)
    train_ds = DocTamperDataset(lmdb_path, minq=args.minq)
    use_pw = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_pw,
        prefetch_factor=4 if use_pw else None,
    )

    eval_sets = [
        ('DocTamperV1-TestingSet',
         os.path.join(args.data_root, 'DocTamperV1-TestingSet')),
        ('DocTamperV1-FCD',
         os.path.join(args.data_root, 'DocTamperV1-FCD')),
        ('DocTamperV1-SCD',
         os.path.join(args.data_root, 'DocTamperV1-SCD')),
    ]

    # ---- model ----
    teacher, student, device = build_models(args)
    criterion = build_criterion(args).to(device)

    trainable = filter(lambda p: p.requires_grad, student.parameters())
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(trainable, lr=args.lr)

    scaler = GradScaler()

    scheduler = None
    if args.eta_min > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min)

    start_epoch = 1
    best_iou = 0.0

    # ---- resume ----
    resume_path = args.resume
    if not resume_path:
        auto = os.path.join(args.save_dir, 'latest.pth')
        if os.path.isfile(auto):
            resume_path = auto
    if resume_path and os.path.isfile(resume_path):
        prev_epoch, best_iou = load_checkpoint(
            resume_path, student, optimizer, scaler, scheduler,
            reset_optimizer=args.reset_optimizer)
        start_epoch = prev_epoch + 1
        if args.reset_optimizer:
            start_epoch = 1
            best_iou = 0.0
            logger.info(f'Loaded weights from {resume_path}, '
                         f'optimizer/scheduler/epoch reset for fine-tuning')
        else:
            logger.info(f'Resumed from {resume_path}, epoch {prev_epoch}, '
                         f'best_iou {best_iou:.4f}')

    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f'Student params: {total_params:,}')
    logger.info(f'Training {args.epochs} epochs, start={start_epoch}')
    if scheduler:
        logger.info(f'Scheduler: CosineAnnealingLR, '
                     f'lr={args.lr} -> eta_min={args.eta_min}')

    # ---- train loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]['lr']
        t0 = time.time()
        train_metrics = train_one_epoch(
            epoch, teacher, student, train_loader, criterion, optimizer,
            scaler, device)
        train_time = time.time() - t0

        if scheduler:
            scheduler.step()

        logger.info(
            f'Epoch {epoch}/{args.epochs} [{train_time:.0f}s] lr={cur_lr:.2e} '
            f'loss={train_metrics["loss"]:.4f} hard={train_metrics["hard"]:.4f} '
            f'soft={train_metrics["soft"]:.4f} feat={train_metrics["feat"]:.4f}')

        # ---- eval on 3 test sets ----
        eval_results = {}
        for ds_name, ds_path in eval_sets:
            res = evaluate(
                student, ds_name, ds_path, args.minq, device,
                batch_size=args.eval_batch_size, num_workers=2,
                base_dir=base_dir)
            eval_results[ds_name] = res
            logger.info(
                f'  [{ds_name}] mIoU={res["mean_iou"]:.4f} '
                f'iou_t={res["iou_tamper"]:.4f} '
                f'P={res["precision"]:.4f} R={res["recall"]:.4f} '
                f'F1={res["f1"]:.4f}')

        # ---- metrics jsonl ----
        record = {
            'epoch': epoch,
            'lr': cur_lr,
            'train': train_metrics,
            'eval': eval_results,
            'time_s': train_time,
        }
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # ---- checkpoint: best (by TestingSet mean_iou) ----
        cur_iou = eval_results['DocTamperV1-TestingSet']['mean_iou']
        if cur_iou > best_iou:
            best_iou = cur_iou
            save_checkpoint(
                os.path.join(args.save_dir, 'best.pth'),
                epoch, student, optimizer, scaler, best_iou, scheduler)
            logger.info(f'  New best mIoU={best_iou:.4f}, saved best.pth')

        # ---- checkpoint: latest (every epoch, after best_iou updated) ----
        save_checkpoint(
            os.path.join(args.save_dir, 'latest.pth'),
            epoch, student, optimizer, scaler, best_iou, scheduler)

        # ---- checkpoint: periodic ----
        if epoch % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.save_dir, f'epoch_{epoch}.pth'),
                epoch, student, optimizer, scaler, best_iou, scheduler)
            logger.info(f'  Saved epoch_{epoch}.pth')

    logger.info(f'Training finished. Best TestingSet mIoU={best_iou:.4f}')


if __name__ == '__main__':
    main()
