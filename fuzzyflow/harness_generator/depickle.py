import pickle
import dace
import numpy as np
import copy

#from fuzzyflow.harness_generator import sdfg2cpp

path_prefix = '.testdata_void/fails/'


pre = dace.SDFG.from_file(path_prefix + 'pre.sdfg')
post = dace.SDFG.from_file(path_prefix + 'post.sdfg')

'''
NPROMA = KLON = np.random.randint(2, 10)
NCLV = np.random.randint(10, 128)
KLEV = 137

KFLDX = np.random.randint(1)
NCLDQI = np.random.randint(1)
NCLDQL = np.random.randint(1)
NCLDQT = np.random.randint(1)
NCLDQR = np.random.randint(1)
NCLDQS = np.random.randint(1)
NCLDQV = np.random.randint(1)

_for_it_24 = np.random.randint(1, KLEV)

ZFALLSINK = np.random.rand(KLON, NCLV)
ZQXN = np.random.rand(KLON, NCLV)
ZRDTGDP = np.random.rand(KLON)
ZPFPLSX = np.random.rand(KLON, KLEV + 1, NCLV)

symbols = {
    'NPROMA': NPROMA,
    'KLON': KLON,
    'NCLV': NCLV,
    'KLEV': KLEV,
    'KFLDX': KFLDX,
    'NCLDQI': NCLDQI,
    'NCLDQL': NCLDQL,
    'NCLDQT': NCLDQT,
    'NCLDQS': NCLDQS,
    'NCLDQV': NCLDQV,
    'NCLDQR': NCLDQR,
    '_for_it_24': _for_it_24,
}

inputs = {
    'ZFALLSINK': ZFALLSINK,
    'ZQXN': ZQXN,
    'ZRDTGDP': ZRDTGDP,
    'ZPFPLSX': ZPFPLSX,
}

input_copies = copy.deepcopy(inputs)
'''

#with open(path_prefix + 'symbols', 'wb') as f:
#    symbols = {
#        'B': 6,
#        'H': 16,
#        'P': 64,
#        'SM': 64,
#        'N': 64,
#        'emb': 64,
#    }
#    pickle.dump(symbols, f, pickle.HIGHEST_PROTOCOL)

inputs = None
symbols = None
constraints = None
with open(path_prefix + 'inputs', 'rb') as f:
    inputs = pickle.load(f)
with open(path_prefix + 'constraints', 'rb') as f:
    constraints = pickle.load(f)
with open(path_prefix + 'symbols', 'rb') as f:
    symbols = pickle.load(f)

input_copies = copy.deepcopy(inputs)

with dace.config.temporary_config():
    dace.config.Config.set('compiler', 'cpu', 'args', value='-std=c++14 -O0 -Wall -fPIC -Wno-unused-parameter -Wno-unused-label -fopenmp')

    pre(**inputs, **symbols)
    post(**input_copies, **symbols)
    valid = True

    for k in inputs.keys():
        oval = inputs[k]
        nval = input_copies[k]

        if isinstance(oval, np.ndarray):
            if not np.allclose(oval, nval):
                print('Mismatch in', k)
                valid = False
        else:
            if oval != nval:
                print('Mismatch in', k)
                valid = False

print('Valid' if valid else 'INVALID!')

'''
init_args = {}
for k in inputs.keys():
    init_args[k] = 'rand'

sdfg2cpp.dump_args('c++', path_prefix + 'harness', init_args, constraints,
                   pre, post,
                   **inputs, **symbols)
                   '''
