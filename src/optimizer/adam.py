from torch import optim
from lighter.decorator import context, reference


class Adam(object):
    @context
    @reference(name='model')
    def __init__(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.settings.bc.lr)

    def __new__(cls, *args, **kwargs):
        obj = super(Adam, cls).__new__(cls, *args, **kwargs)
        obj.__init__()
        return obj.optimizer
