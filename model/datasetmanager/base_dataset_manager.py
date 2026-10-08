class BaseDatasetManager():
    def __init__(self):
        raise NotImplementedError()

    def mask_predict(self,model):
        raise NotImplementedError()
    
    def depth_predict(self,model):
        raise NotImplementedError()
    
    def get_dpdata(self):
        raise NotImplementedError()