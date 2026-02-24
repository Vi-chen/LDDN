import argparse
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset.Transforms as myTransforms
import dataset.dataset as myDataset
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(args):
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    train_trans = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.RandomFlip(),
            myTransforms.RandomExchange(),
            myTransforms.ToTensor(),
        ]
    )
    val_trans = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.ToTensor(),
        ]
    )

    train_set = myDataset.Dataset("train", args.datapath, transform=train_trans)
    val_set = myDataset.Dataset("val", args.datapath, transform=val_trans)

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batchsize,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batchsize,
        num_workers=4,
        shuffle=False,
    )
    return train_loader, val_loader


def compute_loss_and_pred(logits, label_long):
    if logits.shape[-2:] != label_long.shape[-2:]:
        logits = F.interpolate(logits, size=label_long.shape[-2:], mode="bilinear", align_corners=False)
    loss = nn.CrossEntropyLoss()(logits, label_long)
    pred = torch.argmax(logits, dim=1)
    return loss, pred


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0

    for img, label, _ in tqdm(loader):
        x1 = img[:, 0:3].to(device).float()
        x2 = img[:, 3:6].to(device).float()
        y = label.squeeze(1).to(device).long()

        optimizer.zero_grad()
        logits = model(x1, x2)
        loss, _ = compute_loss_and_pred(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach().cpu().item()

    return loss_sum / max(1, len(loader))


def validate(model, loader, device, num_class):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=num_class)
    loss_sum = 0.0

    with torch.no_grad():
        for img, label, _ in tqdm(loader):
            x1 = img[:, 0:3].to(device).float()
            x2 = img[:, 3:6].to(device).float()
            y = label.squeeze(1).to(device).long()

            logits = model(x1, x2)
            loss, pred = compute_loss_and_pred(logits, y)

            loss_sum += loss.detach().cpu().item()
            meter.update_cm(pr=pred.cpu().numpy(), gt=y.cpu().numpy())

    scores = meter.get_scores()
    avg_loss = loss_sum / max(1, len(loader))
    return avg_loss, scores


def run(args):
    set_seed(args.seed)

    os.makedirs(args.logpath, exist_ok=True)
    for fname in ("train.py", "test.py"):
        src = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.logpath, fname))

    writer = SummaryWriter(log_dir=args.logpath)
    train_loader, val_loader = build_dataloaders(args)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    model = ChangeClassifier(
        num_classes=args.num_class,
        num=args.fuse_block,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-4)

    best_f1 = -1.0
    for ep in range(args.epoch):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, scores = validate(model, val_loader, device, args.num_class)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, ep)
        writer.add_scalar("Loss/val", val_loss, ep)
        writer.add_scalar("Metric/F1_1", scores["F1_1"], ep)

        print(
            f"Epoch {ep}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"acc={scores['acc']:.6f} miou={scores['miou']:.6f} F1_1={scores['F1_1']:.6f}"
        )

        if scores["F1_1"] > best_f1:
            best_f1 = scores["F1_1"]
            ckpt = os.path.join(args.logpath, f"best_ep{ep}_f1_{best_f1:.6f}.pth")
            torch.save(model.state_dict(), ckpt)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ours model (minimal pipeline).")
    parser.add_argument("--datapath", default="/userA02/DataSets/WHU-CD-256", type=str)
    parser.add_argument("--logpath", default="./runs/minimal", type=str)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--val_batchsize", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    run(args)
