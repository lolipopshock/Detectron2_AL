import numpy as np


class DefaultSchedular(object):

    def __init__(self, start, end, steps, mode):

        if mode == 'linear':
            # at = a0 + (t-1) * d
            self._val = np.linspace(start, end, steps)
        elif mode == 'geometry':
            # at = a0 * q ** (t-1)
            q = (end/start)**(1/(steps-1))
            self._val = np.array([start*q**i for i in range(steps)])
        else:
            raise NotImplementedError
        
    def __getitem__(self, idx):

        return self._val[idx]


class IntegerSchedular(DefaultSchedular):

    def __init__(self, start, end, steps, mode):

        super().__init__(start, end, steps, mode)

        self._val = self._val.astype('int')