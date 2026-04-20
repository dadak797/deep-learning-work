### Train

```
python main.py --mode train (--epochs 2)
```

### Train with previous weights

```
python main.py --mode train (--epochs 2) (--restart)
```

### Test

```
python main.py --mode test
```

### Train (Test per epoch)
```
python main.py --mode train_and_test (--epoch 2) (--restart) (--device cuda) (--data_aug) (--batch_size 4)
python main.py --mode train_and_test --epoch 50 --device cuda --data_aug --batch_size 4
```

### Resource Monitoring
```
.\.venv\Scripts\activate
pip install nvitop
nvitop
```
