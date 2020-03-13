from lighter.context import Context
from lighter.decorator import context

from src.data.dataset import DataGenerator


class Demo(object):
    @context
    def __init__(self):
        self.generator = DataGenerator()

    def run(self):
        self.generator.init()
        for _ in range(1):
            obs, actions, rewards, next_states, dones = self.generator.sample()
            print(obs.shape, actions['move'].shape, rewards.shape)


if __name__ == "__main__":
    Context.create(device='cpu', config_file='configs/meta.json')
    demo = Demo()
    demo.run()
