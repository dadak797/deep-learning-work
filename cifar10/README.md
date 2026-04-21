# Train

```
python main.py --mode train --epochs 2 --device mps
```

# Train with previous weights

```
python main.py --mode train --epochs 2 --restart --device mps
```

# Test

```
python main.py --mode test --device mps
```

- Training should be done and the weightings should be saved in `saved_models/cifar_net.pth`

# Train and Test

- The model is tested at the end of every epoch

```
python main.py --mode train_and_test --epochs 2 --restart --device mps --data_aug --batch_size 4
python main.py --mode train_and_test --epochs 50 --device mps --data_aug --batch_size 4
```

- `--mode`: train, test, train_and_test
- `--epochs`: Number of epochs
- `--restart`: Start training with saved weights
- `--device`: cpu, cuda, mps(for Mac)
- `--data_aug`: Data augmentation (Horizontal flip and crop)
- `--batch_size`: Batch size

# Resource Monitoring

### Windows

```
.\.venv\Scripts\activate
pip install nvitop
nvitop
```

### Mac

```
pip3 install asitop
sudo asitop
```
