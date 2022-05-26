from .pointPWC import PointPWC

def get_model(name):
    if name == 'pointPWC':
        model = PointPWC()
    else: 
        raise NotImplementedError
    
    return model