import numpy as np

class Msgs:
    def __init__(self):
        self.reset()

    def store(self, name, value):
        self.buffer[name] = value

    def get(self, name):
        if name in self.buffer:
            return self.buffer[name]
        return None

    def reset(self):
        self.buffer = dict()

    def print(self):
        for name, var in self.buffer.items():
            print('-' * 24)
            print(name, ':', var)


global_msgs = Msgs()


def msgProcesser(name, value):
    global_msgs.store(name, value)


def receiver(name, func=msgProcesser):
    def new_func(value):
        func(name, value)

    return new_func


def mapProcesser(name, value):
    '''
    将数据处理成一个矩阵（未知:-1，可通行:0，不可通行:1）
    '''
    width = value.info.width
    height = value.info.height
    resolution = value.info.resolution

    mapdata = np.array(value.data, dtype=np.double)
    value = mapdata.reshape((height, width))
    value /= 100
    # value = value[:, ::-1]

    msgProcesser(name, value)
