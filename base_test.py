import os
from alive_progress import alive_bar

import dace
from dace.sdfg import SDFG
from dace.transformation import dataflow, interstate, subgraph

import fuzzyflow as ff


def _test_from_basedir(datadir: str, base_sdfg: SDFG, category_name: str):
    verify_dict = dict()
    for _, testdirs, _ in os.walk(datadir):
        bartitle = f'Testing {category_name}'
        with alive_bar(len(testdirs), title=bartitle) as bar:
            for dir in testdirs:
                bar.text(os.path.basename(dir))

                # Ignore z_ prefixed directories.
                if dir.startswith('z_'):
                    bar()
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
                    sdfg = (
                        SDFG.from_file(os.path.join(subdir, override_sdfg_name))
                        if override_sdfg_name is not None else base_sdfg
                    )
                    valid = True
                    offending_file = None
                    print('Verifying', subdir)
                    for file in files:
                        if file != override_sdfg_name:
                            xform, _ = ff.load_transformation_from_file(
                                os.path.join(subdir, file), sdfg
                            )

                            verifier = ff.TransformationVerifier(
                                xform, sdfg, ff.CutoutStrategy.SIMPLE
                            )
                            valid = valid and verifier.verify(
                                n_samples=500,
                                debug_save_path=os.path.join(
                                    dbg_save_dir, file.split('.')[0]
                                )
                            )
                            if not valid:
                                offending_file = file
                                break
                    if not valid:
                        print(subdir, 'is invalid!')
                        if offending_file is not None:
                            print('Offending file is', offending_file)
                    else:
                        print(subdir, 'verified')
                    verify_dict[os.path.basename(subdir)] = valid

                bar()

    print('+---------------------------------------+')
    print('| Verification completed                |')
    print('+---------------------------------------+')
    row_format = ('| {:<30} {:>5} |')
    for k in verify_dict:
        print(row_format.format(k, ('🟢' if verify_dict[k] else '❌')))
    print('+---------------------------------------+')


def test_multistate():
    datadir = './tests/data/multistate'
    base_sdfg = SDFG.from_file(os.path.join(datadir, 'summation.sdfg'))
    _test_from_basedir(datadir, base_sdfg, 'multistate transformations')


def test_singlestate():
    datadir = './tests/data/singlestate'
    base_sdfg = SDFG.from_file(os.path.join(datadir, 'hdiff.sdfg'))
    _test_from_basedir(datadir, base_sdfg, 'singlestate transformations')

def main():
    test_singlestate()
    test_multistate()

if __name__ == '__main__':
    main()
