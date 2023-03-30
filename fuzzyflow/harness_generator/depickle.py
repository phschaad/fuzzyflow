import pickle
import dace
import numpy as np

from sdfg2cpp import dump_args

path_prefix = ''


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

print('Inputs:')

init_args = {}
for k in inputs.keys():
    init_args[k] = 'rand'

dump_args('c++', path_prefix + 'harness', init_args, constraints,
                   pre, post,
                   **inputs, **symbols)
