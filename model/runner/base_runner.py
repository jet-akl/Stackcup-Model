class BaseRunner():
    def setup(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()