from common import params
from train import test_process

if __name__ == "__main__":
    print("config:\n", vars(params))
    test_loss = test_process(params)
    print(f"test loss: {test_loss}")
