import argparse
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataloader import DocTamperDataset
from models.dtd import seg_dtd
from models.light_dtd import LightDTD
from models.losses import SoftBCEWithLogitsLoss, LovaszLoss, DiceLoss
from models.losses.distill_loss import DistillationLoss


def _strip_prefix_from_state_dict(state_dict, prefix):
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _load_teacher_state_dict(teacher, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(raw_state)}")

    model_keys = set(teacher.state_dict().keys())
    candidates = [
        raw_state,
        _strip_prefix_from_state_dict(raw_state, 'module.'),
        _strip_prefix_from_state_dict(raw_state, 'model.'),
        _strip_prefix_from_state_dict(_strip_prefix_from_state_dict(raw_state, 'module.'), 'model.'),
    ]

    # 选择与当前模型键重叠最多的候选，避免手工猜前缀
    best_state = max(candidates, key=lambda s: len(model_keys.intersection(s.keys())))
    teacher.load_state_dict(best_state, strict=True)


def _load_student_state_dict(student, raw_state):
    """
    兼容 DataParallel/非 DataParallel 的学生权重加载。
    """
    model_keys = set(student.state_dict().keys())
    candidates = [
        raw_state,
        _strip_prefix_from_state_dict(raw_state, 'module.'),
        _strip_prefix_from_state_dict(raw_state, 'model.'),
        _strip_prefix_from_state_dict(_strip_prefix_from_state_dict(raw_state, 'module.'), 'model.'),
    ]
    best_state = max(candidates, key=lambda s: len(model_keys.intersection(s.keys())))
    student.load_state_dict(best_state, strict=True)


def _patch_legacy_gelu(module):
    """
    兼容旧权重/旧序列化对象中 GELU 缺失 approximate 属性的问题。
    """
    for m in module.modules():
        if isinstance(m, torch.nn.GELU) and not hasattr(m, "approximate"):
            m.approximate = "none"
        # 兼容旧版 timm DropPath 缺失 scale_by_keep 属性
        if m.__class__.__name__ == "DropPath" and not hasattr(m, "scale_by_keep"):
            m.scale_by_keep = True


def _format_teacher_qtable(batch, dct_tensor, device):
    """
    教师 FPH 需要 qtable 形状为 (B, 1, 8, 8)。
    """
    if 'qtb' in batch:
        qtb = batch['qtb'].to(device)
    else:
        qtb = torch.zeros(dct_tensor.size(0), 8, 8, dtype=torch.long, device=device)

    if qtb.dim() == 2:
        # (B, 64) -> (B, 8, 8)
        qtb = qtb.view(qtb.size(0), 8, 8)
    if qtb.dim() == 3:
        # (B, 8, 8) -> (B, 1, 8, 8)
        qtb = qtb.unsqueeze(1)
    return qtb.long()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./', help='根目录，包含 LMDB 与 pks 目录')
    parser.add_argument('--lmdb_name', type=str, default='DocTamperV1-TrainingSet', help='LMDB 文件名')
    parser.add_argument('--minq', type=int, default=75)
    parser.add_argument('--teacher_pth', type=str, default='pths/dtd_doctamper.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='pths')
    parser.add_argument('--save_dir_drive', type=str, default='',
                        help='可选：Google Drive 备份目录（如 /content/drive/MyDrive/DocTamper_ckpts）')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default='', help='断点续训 checkpoint 路径')
    return parser.parse_args()


def build_models(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 教师模型：加载权重并冻结
    teacher = seg_dtd('', 2).to(device)
    _patch_legacy_gelu(teacher)
    _load_teacher_state_dict(teacher, args.teacher_pth)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 学生模型
    student = LightDTD(fph_out_channels=128, pretrained_vph=False, classes=2).to(device)
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
            self.bce = bce
            self.lovasz = lovasz
            self.dice = dice

        def forward(self, logits, target):
            # logits: (N, 2, H, W), target: (N, 1, H, W) 或 (N, H, W)
            if target.dim() == 3:
                target_ = target.unsqueeze(1).float()
            else:
                target_ = target.float()
            target_long = target.squeeze(1).long()
            loss_bce = self.bce(logits[:, 1:2, ...], target_)
            loss_lovasz = self.lovasz(logits, target_long)
            loss_dice = self.dice(logits, target_long)
            return 0.5 * loss_bce + loss_lovasz + loss_dice

    hard_loss = HardLoss()
    distill = DistillationLoss(
        hard_loss_fn=hard_loss,
        temperature=2.0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    return distill


def train_one_epoch(
    epoch,
    teacher,
    student,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
):
    teacher.eval()
    student.train()

    # 抓取教师融合层特征（与学生 adapt 后特征对齐）
    teacher_feat_cache = {'feat': None}
    teacher_fu = teacher.module.model.FU if isinstance(teacher, nn.DataParallel) else teacher.model.FU

    def _save_teacher_feat(_module, _inputs, output):
        teacher_feat_cache['feat'] = output.detach()

    hook_handle = teacher_fu.register_forward_hook(_save_teacher_feat)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', ncols=120)
    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    total_feat = 0.0
    n_batches = 0

    for batch in pbar:
        img_clean = batch['img_clean'].to(device)      # 教师/学生统一输入
        dct_clean = batch['dct_clean'].to(device)      # 教师/学生统一 DCT
        img_dist = img_clean
        dct_dist = dct_clean
        mask = batch['mask'].to(device)                # 真实标签 (N,1,512,512)

        # 教师前向：需要融合中间特征，使用原 DTD 的接口
        with torch.no_grad():
            qt_teacher = _format_teacher_qtable(batch, dct_clean, device)
            teacher_logits = teacher(img_clean, dct_clean, qt_teacher)
            teacher_feat = teacher_feat_cache['feat']
            if teacher_feat is None:
                raise RuntimeError("Failed to capture teacher fusion feature from FU layer.")

        optimizer.zero_grad()
        with autocast():
            # 学生前向：与教师一致，使用同一份 qtable（若缺失则退化为全零）
            if 'qtb' in batch:
                qt_student = batch['qtb'].to(device).view(dct_dist.size(0), -1).long()
            else:
                qt_student = dct_dist.new_zeros(dct_dist.size(0), 64, dtype=torch.long)
            student_logits = student(img_dist, dct_dist, qt_student)
            # 学生中间特征：使用 get_adapted_fusion
            student_feat = student.module.get_adapted_fusion(img_dist, dct_dist, qt_student) if isinstance(student, nn.DataParallel) else student.get_adapted_fusion(img_dist, dct_dist, qt_student)

            loss, loss_hard, loss_soft, loss_feat = criterion(
                student_logits,
                teacher_logits,
                student_feat,
                teacher_feat,
                mask,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n_batches += 1
        total_loss += loss.item()
        total_hard += loss_hard.item()
        total_soft += loss_soft.item()
        total_feat += loss_feat.item()

        pbar.set_postfix({
            'loss': f'{total_loss / n_batches:.4f}',
            'hard': f'{total_hard / n_batches:.4f}',
            'soft': f'{total_soft / n_batches:.4f}',
            'feat': f'{total_feat / n_batches:.4f}',
        })

    hook_handle.remove()

    return (
        total_loss / max(1, n_batches),
        total_hard / max(1, n_batches),
        total_soft / max(1, n_batches),
        total_feat / max(1, n_batches),
    )


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_dir_drive:
        os.makedirs(args.save_dir_drive, exist_ok=True)

    # 数据集和 DataLoader
    lmdb_path = os.path.join(args.data_root, args.lmdb_name)
    train_dataset = DocTamperDataset(lmdb_path, minq=args.minq)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    teacher, student, device = build_models(args)
    criterion = build_criterion(args).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
    scaler = GradScaler()
    start_epoch = 1

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location='cpu')
        student_state = resume_ckpt.get('student_state', resume_ckpt.get('state_dict', None))
        if student_state is None:
            raise KeyError(f"Checkpoint {args.resume} does not contain student_state/state_dict.")
        _load_student_state_dict(student, student_state)
        if 'optimizer' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
        if 'scaler' in resume_ckpt:
            scaler.load_state_dict(resume_ckpt['scaler'])
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        print(f"Resume training from epoch {start_epoch} with checkpoint: {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):
        loss, hard, soft, feat = train_one_epoch(
            epoch,
            teacher,
            student,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        print(f'Epoch {epoch}: loss={loss:.4f}, hard={hard:.4f}, soft={soft:.4f}, feat={feat:.4f}')

        if epoch % args.save_interval == 0:
            ckpt_name = f'light_dtd_distill_epoch{epoch}.pth'
            save_path = os.path.join(args.save_dir, ckpt_name)
            state = {
                'epoch': epoch,
                'student_state': student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }
            torch.save(state, save_path)
            print(f'Saved student checkpoint to {save_path}')
            if args.save_dir_drive:
                drive_save_path = os.path.join(args.save_dir_drive, ckpt_name)
                try:
                    torch.save(state, drive_save_path)
                    print(f'Backed up checkpoint to Google Drive: {drive_save_path}')
                except Exception as exc:
                    warnings.warn(
                        f'Failed to back up checkpoint to Google Drive path {drive_save_path}: {exc}',
                        RuntimeWarning
                    )


if __name__ == '__main__':
    main()

