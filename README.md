# LDDN: A Lightweight Dual-Domain Directional Network for Remote Sensing Change Detection


## Environment

建议使用 Python 3.8+。

安装依赖：

```bash
pip install -r requirements.txt
```

## Dataset Format

`dataset/dataset.py` 支持两种组织方式。

### 1) Folder-split mode

```text
<DATA_ROOT>/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### 2) List mode

```text
<DATA_ROOT>/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

`train.txt/val.txt/test.txt` 中每一行是样本文件名（A/B/label 三个目录同名对应）。

## Train

```bash
python3 train.py 
```
训练过程中会在 `--logpath` 下记录 TensorBoard 日志，并按验证集 `F1_1` 保存最优权重。

## Test

```bash
python3 test.py 
```
