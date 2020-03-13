from lighter.decorator import context, reference


class Evaluation(object):
    @context
    @reference(name='env_wrapper')
    @reference(name='model')
    @reference(name='transform')
    def __init__(self):
        self.env = self.env_wrapper.env

    def run(self):
        obs = self.env.reset()
        dones = self.env.values(False)
        net_rewards = self.env.values(0.0)
        while not all(dones):
            obs = self.transform.prepare_for_model(obs)
            actions = self.model.act(obs)
            actions = self.transform.prepare_for_env(actions, self.env.noops())
            obs, rewards, dones, infos = self.env.step(actions)
            net_rewards += rewards
