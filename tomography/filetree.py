import os
import json
import itertools

class FileTree:

    def __init__(self, analysis_home, config_name='filetree.json'):
        self._home_dir = analysis_home
        with open(os.path.join(analysis_home, config_name), 'r') as fp:
            config = json.load(fp)
        
        self._order = config['order']
        self._parameters = config['parameter']
        self._extension = config['ext']

    @property
    def order(self):
        return self._order

    @property
    def parameters(self):
        return self._parameters

    @property
    def format(self):
        return self.format_omission()

    def format_omission(self, omit_params=None):
        if omit_params is None:
            omit_params = set()

        path_format = {}
        for key, params in self._order.items():
            path_format[key] = self._home_dir
            for param in params:
                if param in omit_params:
                    break
                path_format[key] = os.path.join(path_format[key], f"{{{param}}}")

        return path_format

    @property
    def home_dir(self):
        return self._home_dir

    def parameter_product(self, param_list):
        """ Generate product of undefined parameters.

        Params
        =======
        param_list : Iterable(str)
            List of undefined parameter names.

        =======

        Returns
        =======
        param : Generator(dict)
            Generator of parameter product dictionary.

        =======
        """
        product = itertools.product(*[self._parameters[key] for key in param_list])
        for param in product:
            yield {k: param[i] for i,k in enumerate(param_list)}

    def get_filelist(self, data_type='data', fixed_params=None, extension='', omit_params=None):
        """ Get generator of files selected by parameters.

        Params
        =======
        data_type : str
            Name of selected format.
            Ex. 'data', 'noise'

        fixed_params : dict(*param)
            Dictionary of fixed parameters.
            key is name of parameter and value is parameter (whose type depends on each parameter).
            In default, empty dict {}.
        =======
        
        Returns
        =======
        file_list : Generator(str)
            Generator of file pathes belongs to given parameters.

        =======
        """
        if extension == '':
            extension = self._extension

        order = self._order[data_type]

        if fixed_params is None:
            fixed_params = dict()

        if omit_params is None:
            omit_params = set()

#        if len(fixed_params) > len(order):
#            raise IndexError("Too many parameters")

        if any(not key in self._parameters or not value in set(self._parameters[key]) for key, value in fixed_params.items()):
            print(order, self._parameters, fixed_params)
            raise KeyError("Cannot find key or value in candidates")

        undefined_params = [key for key in order if not key in fixed_params]
        formats = self.format_omission(omit_params)
        for temp_params in self.parameter_product(undefined_params):
            yield formats[data_type].format(**dict(fixed_params, **temp_params)) + extension


if __name__ == '__main__':
    t = FileTree('./20190729_hist2d_rotate/blue_40_gamma_0p06/')
    print(t.order)
    print(t.format)
    print(list(t.get_filelist(fixed_params={'source':"hd1", "phase":0, "FF":"qON"})))
    t.parameters['source']=[]
    print(t.parameters)