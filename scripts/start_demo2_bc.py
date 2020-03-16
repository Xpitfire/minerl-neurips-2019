import minerl
from lighter.context import Context
from lighter.decorator import context


class Demo(object):
    @context
    def __init__(self):
        self.data = minerl.data.make(self.config.settings.env.env,
                                     data_dir=self.config.settings.env.data_root,
                                     force_download=True)
        self.generator = self.data.sarsd_iter(num_epochs=1, max_sequence_len=self.config.settings.model.seq_len)

    def run(self):
        cnt = 0
        try:
            while True:
                next(self.generator)
                cnt += 1
        except Exception as e:
            print(e)
        print('Iterations counted:', cnt)


if __name__ == "__main__":
    Context.create(device='cpu', config_file='configs/meta.json')
    demo = Demo()
    demo.run()
