import os

import dace
from dace.sdfg import SDFG
from dace.transformation import dataflow, interstate, subgraph

import fuzzyflow as ff


def test_singlestate():
    verify_dict = dict()
    datadir = './tests/data/singlestate'
    base_sdfg = SDFG.from_file(os.path.join(datadir, 'hdiff.sdfg'))
    for _, testdirs, _ in os.walk(datadir):
        for dir in testdirs:
            # Ignore z_ prefixed directories.
            if dir.startswith('z_'):
                continue
            for subdir, _, files in os.walk(os.path.join(datadir, dir)):
                # Ignore the debug directory.
                if subdir.endswith('dbg'):
                    continue

                dbg_save_dir = os.path.join(subdir, 'dbg')
                if not os.path.exists(dbg_save_dir):
                    os.makedirs(dbg_save_dir)

                override_sdfg_name = None
                for file in files:
                    if file.endswith('.sdfg'):
                        override_sdfg_name = file
                sdfg = (SDFG.from_file(os.path.join(subdir, override_sdfg_name))
                    if override_sdfg_name is not None else base_sdfg)
                valid = True
                for file in files:
                    if file != override_sdfg_name:
                        xform, _ = ff.load_transformation_from_file(
                            os.path.join(subdir, file), sdfg
                        )

                        verifier = ff.TransformationVerifier(
                            xform, sdfg, ff.CutoutStrategy.SIMPLE
                        )
                        valid = valid and verifier.verify(
                            n_samples=100,
                            debug_save_path=os.path.join(
                                dbg_save_dir, file.split('.')[0]
                            )
                        )
                        if not valid:
                            break
                if not valid:
                    print(subdir, 'is invalid!')
                else:
                    print(subdir, 'verified')
                verify_dict[os.path.basename(subdir)] = valid

    print('+---------------------------------------+')
    print('| Verification completed                |')
    print('+---------------------------------------+')
    row_format = ('| {:<30} {:>5} |')
    for k in verify_dict:
        print(row_format.format(k, ('🟢' if verify_dict[k] else '❌')))
    print('+---------------------------------------+')

def main():
    test_singlestate()

if __name__ == '__main__':
    main()
