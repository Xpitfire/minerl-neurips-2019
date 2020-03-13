import numpy as np
from tqdm import tqdm
from torch import optim
from lighter.decorator import context, reference, device
from src.data.dataset import DataGenerator


class Runner(object):
    @device
    @context
    @reference(name='model')
    @reference(name='transform')
    @reference(name='eval_env')
    @reference(name='bc_criterion')
    @reference(name='collectible')
    @reference(name='writer')
    def __init__(self):
        self.generator = DataGenerator()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.settings.bc.lr)

    def run(self):
        self.generator.init()
        self.model.to(self.device)
        for e in tqdm(range(self.config.settings.bc.epochs)):
            self.model.train()
            for _ in range(self.config.settings.bc.batches):
                batch = self.generator.sample()
                pred = self.model.act(batch['obs'])
                evaluation = self.model.evaluate(batch['obs'], batch['actions'])
                loss = self.bc_criterion(batch, pred, evaluation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.collectible.update(category='train', **{'loss': loss.detach().cpu().item()})
            collection = self.collectible.redux(func=np.mean)
            self.writer.write(category='train', **collection)
            self.writer.step()
            self.collectible.reset()
            self.model.checkpoint(e)
            self.model.eval()
            self.eval_env.run()


