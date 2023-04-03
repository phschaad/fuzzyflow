import pickle
import dace
import numpy as np
import copy

#from fuzzyflow.harness_generator import sdfg2cpp

path_prefix = '.testdata_case_studies/cloudsc/LoopToMap_Stage2/fails/LoopToMap_1/'


pre = dace.SDFG.from_file(path_prefix + 'pre.sdfg')
post = dace.SDFG.from_file(path_prefix + 'post.sdfg')

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
