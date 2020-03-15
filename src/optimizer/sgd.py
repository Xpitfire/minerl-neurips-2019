from torch import optim
from lighter.decorator import context, reference


class SGD(object):
    @context
    @reference(name='model')
    def __init__(self):
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config.settings.bc.lr,
                                   momentum=self.config.settings.bc.momentum)

    def __new__(cls, *args, **kwargs):
        obj = super(SGD, cls).__new__(cls, *args, **kwargs)
        return obj.optimizer
