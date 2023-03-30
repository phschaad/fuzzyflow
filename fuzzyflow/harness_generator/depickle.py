import pickle
import dace
import numpy as np

from fuzzyflow.harness_generator import sdfg2cpp

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

init_args = {}
for k in inputs.keys():
    init_args[k] = 'rand'

sdfg2cpp.dump_args('c++', path_prefix + 'harness', init_args, constraints,
                   pre, post,
                   **inputs, **symbols)
