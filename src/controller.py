from lighter.decorator import context, config


class MetaController(object):
    @context
    @config(path='configs/meta.json')
    def __init__(self):
        pass
