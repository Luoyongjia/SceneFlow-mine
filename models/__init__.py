from .pointPWC import PointPWC
from .scoreNet import ScoreNet

def get_model(name):
    if name == 'pointPWC':
        model = PointPWC()
        return model
    elif name == 'dual':
        model_l = PointPWC()
        model_r = PointPWC()
        model_mn = ScoreNet(16, 3, 64)
        return model_l, model_r, model_mn
    else: 
        raise NotImplementedError
    