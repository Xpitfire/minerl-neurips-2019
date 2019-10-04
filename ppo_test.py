import traceback
from src.ppo.run import run


if __name__ == "__main__":
    try:
        run()
    except:
        traceback.print_exc()
        exit(-1)
