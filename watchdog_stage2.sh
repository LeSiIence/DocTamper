#!/bin/bash
# 训练中断自动续训守护脚本，由 cron 每分钟调用
set -uo pipefail

PROJECT_DIR="/root/autodl-tmp/DocTamper"
cd "$PROJECT_DIR"

LOG_DIR="logs_stage2"
SAVE_DIR="pths_stage2"
PID_FILE="$LOG_DIR/train.pid"
LATEST="$SAVE_DIR/latest.pth"
NOHUP_LOG="$LOG_DIR/nohup_train.log"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"

mkdir -p "$LOG_DIR" "$SAVE_DIR"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" >> "$WATCHDOG_LOG"; }

# 进程还活着就退出
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    exit 0
fi

# 检查 GPU 是否可用
if ! nvidia-smi > /dev/null 2>&1; then
    log "No GPU available, skip."
    exit 0
fi

# 判断是首次启动还是续训
if [ -f "$LATEST" ]; then
    RESUME_ARGS="--resume $LATEST"
    log "Detected interrupted training, resuming from $LATEST"
else
    RESUME_ARGS="--resume pths/best.pth --reset_optimizer"
    log "First launch, starting from pths/best.pth with optimizer reset"
fi

nohup python -u train_distill.py \
    --data_root ./ \
    --lmdb_name DocTamperV1-TrainingSet \
    --minq 75 \
    --teacher_pth pths/dtd_doctamper.pth \
    $RESUME_ARGS \
    --epochs 30 \
    --batch_size 48 \
    --eval_batch_size 8 \
    --num_workers 8 \
    --lr 5e-5 \
    --eta_min 1e-6 \
    --weight_decay 1e-4 \
    --alpha 1.0 \
    --beta 0.5 \
    --gamma 0.1 \
    --save_dir "$SAVE_DIR" \
    --save_interval 5 \
    --log_dir "$LOG_DIR" \
    >> "$NOHUP_LOG" 2>&1 &

echo $! > "$PID_FILE"
log "Training started (PID $(cat "$PID_FILE"))"
