from lighter.context import Context
from lighter.decorator import context
from src.data.dataset import DataGenerator


class Demo(object):
    @context
    def __init__(self):
        pass

    def run(self):
        cnt = 0
        for sample in DataGenerator():
            cnt += 1
        print(cnt)


if __name__ == "__main__":
    Context.create(device='cpu', config_file='configs/meta.json')
    demo = Demo()
    demo.run()
