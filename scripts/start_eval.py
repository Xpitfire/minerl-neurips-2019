from lighter.context import Context
from lighter.decorator import context, reference, device


class Runner(object):
    @device
    @context
    @reference(name='eval_env')
    @reference(name='model')
    def __init__(self):
        pass

    def run(self):
        self.model.load(self.config.settings.eval.checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.eval_env.run()


if __name__ == "__main__":
    Context.create(device='cpu', config_file='configs/meta.json')
    runner = Runner()
    runner.run()
