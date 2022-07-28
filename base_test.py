from copy import deepcopy
import os

import dace
from dace.transformation import interstate, dataflow, subgraph
from dace.sdfg import SDFG

import fuzzyflow as ff


def test_singlestate():
    datadir = './tests/data/singlestate'
    base_sdfg = SDFG.from_file(os.path.join(datadir, 'hdiff.sdfg'))
    for _, testdirs, _ in os.walk(datadir):
        for dir in testdirs:
            for subdir, _, files in os.walk(os.path.join(datadir, dir)):
                override_sdfg_name = None
                for file in files:
                    if file.endswith('.sdfg'):
                        override_sdfg_name = file
                sdfg = (SDFG.from_file(os.path.join(subdir, override_sdfg_name))
                    if override_sdfg_name is not None else base_sdfg)
                for file in files:
                    if file != override_sdfg_name:
                        xform, _ = ff.load_transformation_from_file(
                            os.path.join(subdir, file), sdfg
                        )

                        verifier = ff.TransformationVerifier(
                            xform, sdfg, ff.CutoutStrategy.SIMPLE
                        )
                        valid = verifier.verify(10)
                        if not valid:
                            print(subdir, 'is invalid!')
                        else:
                            print(subdir, 'verified')


def main():
    test_singlestate()

if __name__ == '__main__':
    main()
