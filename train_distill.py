import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataloader import DocTamperDataset
from models.dtd import seg_dtd
from models.light_dtd import LightDTD
from models.losses import SoftBCEWithLogitsLoss, LovaszLoss
from models.losses.distill_loss import DistillationLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./', help='根目录，包含 LMDB 与 pks 目录')
    parser.add_argument('--lmdb_name', type=str, default='DocTamperV1-FCD', help='LMDB 文件名')
    parser.add_argument('--minq', type=int, default=75)
    parser.add_argument('--teacher_pth', type=str, default='pths/dtd_doctamper.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='pths')
    parser.add_argument('--save_interval', type=int, default=5)
    return parser.parse_args()


def build_models(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 教师模型：加载权重并冻结
    teacher = seg_dtd('', 2).to(device)
    ckpt = torch.load(args.teacher_pth, map_location='cpu')
    teacher.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 学生模型
    student = LightDTD(fph_out_channels=128, pretrained_vph=False, classes=2).to(device)
    student.train()

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    return teacher, student, device


def build_criterion(args):
    # BCE + Lovasz 作为 hard loss
    bce = SoftBCEWithLogitsLoss()
    lovasz = LovaszLoss(mode='multiclass')

    class HardLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = bce
            self.lovasz = lovasz

        def forward(self, logits, target):
            # logits: (N, 2, H, W), target: (N, 1, H, W) 或 (N, H, W)
            if target.dim() == 3:
                target_ = target.unsqueeze(1).float()
            else:
                target_ = target.float()
            loss_bce = self.bce(logits[:, 1:2, ...], target_)
            # Lovasz 接收 logits 与 [B,H,W] 的标签（前景类索引 1）
            loss_lovasz = self.lovasz(logits, target.squeeze(1).long())
            return loss_bce + loss_lovasz

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

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', ncols=120)
    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    total_feat = 0.0
    n_batches = 0

    for batch in pbar:
        img_clean = batch['img_clean'].to(device)      # 给教师
        img_dist = batch['img_dist'].to(device)        # 给学生
        dct_clean = batch['dct_clean'].to(device)      # 教师 DCT
        dct_dist = batch['dct_dist'].to(device)        # 学生 DCT
        mask = batch['mask'].to(device)                # 真实标签 (N,1,512,512)

        # 教师前向：需要融合中间特征，使用原 DTD 的接口
        with torch.no_grad():
            teacher_logits = teacher(img_clean, dct_clean, batch.get('qtb', None).to(device) if 'qtb' in batch else dct_clean.new_zeros(dct_clean.size(0), 64, dtype=torch.long))
            # 教师中间特征：这里假设使用 decoder 之前的融合特征，需从 seg_dtd.model 中获取
            # 简单起见，可先使用最终 logits 作为 feature proxy
            teacher_feat = teacher_logits

        optimizer.zero_grad()
        with autocast():
            # 学生前向
            # qtable 这里没有 LMDB 中的显式 qtb，可用全零或根据需要扩展 dataloader
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

    return (
        total_loss / max(1, n_batches),
        total_hard / max(1, n_batches),
        total_soft / max(1, n_batches),
        total_feat / max(1, n_batches),
    )


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

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

    for epoch in range(1, args.epochs + 1):
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
            save_path = os.path.join(args.save_dir, f'light_dtd_distill_epoch{epoch}.pth')
            state = {
                'epoch': epoch,
                'student_state': student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, save_path)
            print(f'Saved student checkpoint to {save_path}')


if __name__ == '__main__':
    main()

