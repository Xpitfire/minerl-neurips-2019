from lighter.context import Context
from src.bc.runner import Runner


if __name__ == "__main__":
    Context.create(device='cuda:0', config_file='configs/meta.json')
    runner = Runner()
    runner.run()
