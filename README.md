# Experimental Results

---

## Supervised Learning

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method supervised_learning
```

### Training log
```bash
epoch 1/10 | train loss 1.9265
epoch 2/10 | train loss 1.6856
...
epoch 10/10 | train loss 1.0574
```
### Final Test Accuracy: 69.39%


## Rotnet

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method rotnet
```

### Training log
```bash
epoch 1/10 | train loss 1.2019
epoch 2/10 | train loss 1.1233
...
epoch 10/10 | train loss 0.8340
```
### Final Rotation Test Accuracy: 65.87%


## SimCLR

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method simclr
```

### Training log
```bash
epoch 1/10 | train loss 4.7659
epoch 2/10 | train loss 4.4818
...
epoch 10/10 | train loss 4.1559
```
### Final kNN Val Accuracy(k=1): 52.50%



## MoCo-v2

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method moco_v2
```

### Training log
```bash
epoch 1/10 | train loss 8.0228
epoch 2/10 | train loss 7.7930
...
epoch 10/10 | train loss 7.2434
```
### Final kNN Val Accuracy(k=1): 40.75%



## BYOL

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method byol --proj_hidden_dim 4096 --proj_output_dim 256
```

### Training log
```bash
epoch 1/10 | train loss -0.7051
epoch 2/10 | train loss -0.7961
...
epoch 10/10 | train loss -0.8503
```
### Final kNN Val Accuracy(k=1): 53.30%



## SimSiam

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method simsiam
```

### Training log
```bash
epoch 1/10 | train loss -0.6693
epoch 2/10 | train loss -0.6740
...
epoch 10/10 | train loss -0.7901
```
### Final kNN Val Accuracy(k=1): 47.36%
