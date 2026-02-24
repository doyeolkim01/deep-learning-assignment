# Experimental Results

---

## Supervised Learning

### Command

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method supervised_learning
```

Training log
epoch 1/10 | train loss 1.9265
epoch 2/10 | train loss 1.6856
...
epoch 10/10 | train loss 1.0574

Final Test Accuracy: 69.39%

```bash
python main.py --dataset cifar10 --model resnet20 --lr 0.1 --batch_size 128 --method rotnet

epoch 1/10 | train loss 1.2019
epoch 2/10 | train loss 1.1233
...
epoch 10/10 | train loss 0.8340

Final Rotation Test Accuracy: 65.87%
