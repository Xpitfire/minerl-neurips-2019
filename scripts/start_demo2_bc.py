import minerl
from lighter.config import Config


class Demo(object):
    def __init__(self, config):
        self.config = config
        self.data = minerl.data.make(self.config.settings.env.env,
                                     data_dir=self.config.settings.env.data_root,
                                     force_download=True)
        self.generator = self.data.sarsd_iter(num_epochs=1, max_sequence_len=self.config.settings.model.seq_len)

    def run(self):
        cnt = 0
        try:
            while True:
                state, action, reward, next_state, done = next(self.generator)
                cnt += 1
        except Exception as e:
            print(e)
        print('Iterations counted:', cnt)


if __name__ == "__main__":
    config = Config.create_instance(config_dict={
        "settings": {
            "env": "config::configs/env_treechop.json",
            "model": "config::configs/model_discrete_15.json",
            "bc": "config::configs/bc_discrete.json",
            "eval": "config::configs/eval.json"
        }
    })
    demo = Demo(config)
    demo.run()
