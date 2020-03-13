from lighter.context import Context
from lighter.decorator import context, reference


class Demo(object):
    @context
    @reference(name='env_wrapper')
    def __init__(self):
        self.env = self.env_wrapper.env

    def demo_action(self):
        action = self.env.action_space.noop()
        action['camera'] = [0, 0]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1
        return action

    def run(self):
        net_rewards = self.env.values(0.0)
        for _ in range(70):
            actions = self.env.values_lambda(self.demo_action)
            obs, rewards, dones, infos = self.env.step(actions)

            net_rewards += rewards
            print("Total reward: ", net_rewards)

        self.env.close()


if __name__ == "__main__":
    Context.create(device='cpu', config_file='configs/meta.json')
    demo = Demo()
    demo.run()
