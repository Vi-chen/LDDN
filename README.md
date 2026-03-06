# LDDN: A Lightweight Dual-Domain Directional Network for Remote Sensing Change Detection


## Environment

Suggestion to use Python 3.8+。

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

`dataset/dataset.py` Two organizational methods are supported.

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

Each line in `train.txt`, `val.txt` and `test.txt` is the file name of the sample (the names in the A/B/label directories are the same).

## Train

```bash
python3 train.py 
```
During the training process, TensorBoard logs will be recorded under `--logpath`, and the best weights will be saved based on the validation set `F1_1`.

## Test

```bash
python3 test.py 
```
