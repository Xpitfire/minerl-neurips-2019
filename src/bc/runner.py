import numpy as np
import torch
from tqdm import tqdm
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
    @reference(name='optimizer')
    def __init__(self):
        self.generator = DataGenerator()

    def run(self):
        self.model.load(self.config.settings.bc.checkpoint)
        self.model.to(self.device)
        for e in tqdm(range(self.config.settings.bc.epochs)):
            self.model.train()
            for batch in self.generator:
                pred = self.model.act(batch['obs'])
                loss = self.bc_criterion(pred, batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.settings.bc.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.settings.bc.gradient_clipping)
                self.optimizer.step()
            collection = self.collectible.redux(func=np.mean)
            self.writer.write(category='train', **collection)
            self.writer.step()
            self.collectible.reset()
            self.model.checkpoint(e)
            self.model.eval()
            self.eval_env.run()
