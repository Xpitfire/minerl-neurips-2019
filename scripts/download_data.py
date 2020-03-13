import minerl
from lighter.config import Config

if __name__ == "__main__":
    config = Config(path='configs/env.json')
    minerl.data.download(directory=config.data_root)
