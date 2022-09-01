import numpy as np
import os
import itertools
import json


class Quadrature:
    """ Data container of quadratures with save/load function.
    """

    def __init__(self, name, data_iter, hbar=1, **option):
        self._name = name
        self._data = data_iter
        self.HBAR = hbar
        self._option = option

    @property
    def data(self):
        """ Quadrature data with numpy 1d-array format.
        """
        self._data, result = itertools.tee(self._data)
        return np.fromiter(result, float)

    @property
    def name(self):
        return self._name

    def save(self, dirname, file_prefix=""):
        """ Save quadrature array as a .npy file.
        HBAR and quadrature name are also saved as a .json file in the same directory.

        Params
        =======
            dirname : str
                File path to be saved at.

            file_prefix="" : str (optional)
                Saved file name. If empty, the name of quadrature class is used.
        
        =======
        """
        if file_prefix == "":
            file_prefix = self.name
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, f"{file_prefix}.npy"), self.data)
        with open(os.path.join(dirname, f"{file_prefix}_config.json"), "w") as fp:
            json.dump({"name":self.name, "hbar":self.HBAR, 'option': self._option}, fp)

    @staticmethod
    def load(dirname, file_prefix):
        with open(os.path.join(dirname, f"{file_prefix}_config.json"), "r") as fp:
            config = json.load(fp)
        return Quadrature(config['name'], np.load(os.path.join(dirname, f"{file_prefix}.npy")), config['hbar'], **config['option'])

class QuadratureConverter:
    
    @staticmethod
    def convert(data, sn, cn, mode_impulse=np.array([1]), hbar=1, timings=None):
        """ Convert waves to quadratures.

        Params
        =======
        data : Iterable(numpy 1d-array)
            Curves of measured quadratures.

        sn : Iterable(numpy 1d-array)
            Curves of shot-noise.

        cn : Iterable(numpy 1d-array)
            Curves of circuit-noise.

        mode_impulse : numpy 1d-array
            Reversal of mode function.

        hbar : float
            hbar for generated quadrature.

        timings : list(float) [s]
            Picking-up timings. Zero is defined as the first data point.
        
        =======
        """

        data_picked = QuadratureConverter.pickup_trigger(data, mode_impulse, timings)
        sn_picked = QuadratureConverter.pickup_trigger(sn, mode_impulse, timings)
        cn_picked = QuadratureConverter.pickup_trigger(cn, mode_impulse, timings)

        return Quadrature(data.name, QuadratureConverter.normalize(data_picked, sn_picked, cn_picked, mode_impulse, hbar), hbar)

    @staticmethod
    def normalize(data, sn, cn, mode_impulse=np.array([1]), hbar=1):
        """ Convert voltages to quadratures.
        Elements of data, sn, and cn, and mode_impulse must have the same size.

        Params
        =======
        data : Iterable(numpy 1d-array)
            Curves of measured quadratures.

        sn : Iterable(numpy 1d-array)
            Curves of shot-noise.

        cn : Iterable(numpy 1d-array)
            Curves of circuit-noise.

        mode_impulse : numpy 1d-array
            Reversal of mode function.

        hbar : float
            hbar for generated quadrature.

        =======
        """

        filtering = lambda v: np.sum(v*mode_impulse[::-1])

        sn_filtered = list(map(filtering, sn))
        cn_filtered = list(map(filtering, cn))

        sn_ave = np.average(sn_filtered)
        sn_var = np.var(sn_filtered)
        cn_var = np.var(cn_filtered)

        for vol_filtered in map(filtering, data):
            yield (vol_filtered - sn_ave) / np.sqrt(sn_var - cn_var) * np.sqrt(hbar / 2)


    @staticmethod
    def pickup_trigger(wave, mode_impulse=np.array([1]), timings=None):
        """ Pickup arrival timings.

        Params
        =======
        wave : Waveform
            Waveform data.
            
        timings : list(float) [s]
            Picking-up timings. Zero is defined as the first data point.

        =======

        Returns
        =======
        result : gen(1-d array slices)
            Sliced waves.

        =======
        """
        if timings is None:
            index = np.full(wave.config['frame_count'], int(wave.config['data_points']-1))
        elif not hasattr(timings, '__iter__'):
            index = np.full(wave.config['frame_count'], int(timings / wave.config['sampling_interval']))
        elif len(timings) != wave.config['frame_count']:
            print(f"Warning : Timing data does not have the same size as {wave.name}. \
                     Instead used averaged timings.")
            index = np.full(wave.config['frame_count'], int(np.average(timings) / wave.config['sampling_interval']))
        else:
            index = np.array([t / wave.config['sampling_interval'] for t in timings], dtype=int)
        #print(index)

        picked = (q[index[i]-len(mode_impulse)+1:index[i]+1] for i,q in enumerate(wave.voltages))

        return picked
