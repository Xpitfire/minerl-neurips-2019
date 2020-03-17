from lighter.config import Config
from pyvirtualdisplay import Display
pydisplay = Display(visible=0, size=(640, 480))
pydisplay.start()
from src.envs.env_server import start


if __name__ == "__main__":
    args = Config(path='configs/env_diamond.json')
    start(args)
