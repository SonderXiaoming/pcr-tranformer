from common import params
from train import use_process

if __name__ == "__main__":
    print("config:\n", vars(params))
    output = use_process(params)
    print(output)
