import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset.Transforms as myTransforms
import dataset.dataset as myDataset
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier


def build_test_loader(args):
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    test_trans = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.ToTensor(),
        ]
    )
    dataset = myDataset.Dataset("test", args.datapath, transform=test_trans)
    return DataLoader(dataset, batch_size=args.test_batchsize, shuffle=False)


def evaluate(model, loader, device, num_class):
    meter = ConfuseMatrixMeter(n_class=num_class)
    model.eval()

    with torch.no_grad():
        for img, label, _ in tqdm(loader):
            x1 = img[:, 0:3].to(device).float()
            x2 = img[:, 3:6].to(device).float()
            y = label.squeeze(1).to(device).long()

            logits = model(x1, x2)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

            pred = torch.argmax(logits, dim=1)
            meter.update_cm(pr=pred.cpu().numpy(), gt=y.cpu().numpy())

    return meter.get_scores()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    model = ChangeClassifier(
        num_classes=args.num_class,
        num=args.fuse_block,
        pretrained=False,
    )
    state = torch.load(args.modelpath, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    test_loader = build_test_loader(args)
    scores = evaluate(model, test_loader, device, args.num_class)

    print(
        "F1_1 = {F1_1:.6f}, IoU_1 = {iou_1:.6f}, Precision_1 = {precision_1:.6f}, "
        "Recall_1 = {recall_1:.6f}, Acc = {acc:.6f}, mIoU = {miou:.6f}".format(**scores)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ours model (minimal pipeline).")
    parser.add_argument("--datapath", default="/userA02/DataSets/SYSU-CD-256", type=str)
    parser.add_argument("--modelpath", required=True, type=str)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--test_batchsize", type=int, default=1)
    args = parser.parse_args()

    run(args)
