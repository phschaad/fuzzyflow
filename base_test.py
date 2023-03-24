import os
from typing import Dict, List
from alive_progress import alive_bar

from dace.sdfg import SDFG

import fuzzyflow as ff


def _test_from_basedir(datadir: str, category_name: str):
    verify_dict = dict()
    testdirs = os.listdir(datadir)
    bartitle = f'Testing {category_name}'
    with alive_bar(len(testdirs), title=bartitle) as bar:
        for dir in testdirs:
            bar.text(os.path.basename(dir))

            # Ignore z_ prefixed directories.
            if dir.startswith('z_'):
                bar()
                continue

            valid = True
            offending_file = None
            print('Verifying', dir)

            testdir = os.path.join(datadir, dir)

            dbg_save_dir = os.path.join(testdir, 'dbg')
            if not os.path.exists(dbg_save_dir):
                os.makedirs(dbg_save_dir)

            test_sdfgs: Dict[str, SDFG] = dict()
            test_files: List[str] = []
            for file in os.listdir(testdir):
                # Ignore the debug directory.
                if file == 'dbg':
                    continue

                if file.endswith('.sdfg'):
                    sdfg = SDFG.from_file(os.path.join(testdir, file))
                    sdfg_name = file.split('.')[0]
                    test_sdfgs[sdfg_name] = sdfg
                else:
                    test_files.append(file)
            for sdfg_name, _ in test_sdfgs.items():
                for file in test_files:
                    if file.startswith(sdfg_name):
                        testfile = os.path.join(testdir, file)
                        xform, _ = ff.load_transformation_from_file(
                            testfile, sdfg
                        )

                        verifier = ff.TransformationVerifier(
                            xform, sdfg, ff.SamplingStrategy.SIMPLE_UNIFORM
                        )
                        valid = valid and verifier.verify(
                            n_samples=100,
                            debug_save_path=os.path.join(
                                dbg_save_dir, file.split('.')[0]
                            ),
                            enforce_finiteness=True
                        )
                        if not valid:
                            offending_file = file
                            break
                if not valid:
                    break

            if not valid:
                print(dir, 'is invalid!')
                if offending_file is not None:
                    print('Offending file is', offending_file)
            else:
                print(dir, 'verified')
            verify_dict[os.path.basename(dir)] = valid

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
    _test_from_basedir(datadir, 'multistate transformations')


def test_singlestate():
    datadir = './tests/data/singlestate'
    _test_from_basedir(datadir, 'singlestate transformations')

def main():
    test_singlestate()
    test_multistate()

if __name__ == '__main__':
    main()
