from common import params
from train import train_process

if __name__ == "__main__":
    print("config:\n", vars(params))
    train_losses, val_losses = train_process(params)
