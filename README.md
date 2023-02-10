# FuzzyFlow

## Prerequisites
A valid installation of [DaCe](https://github.com/spcl/dace) is required.
The minimum supported version is 0.13.1.

This tool requires Python 3.8 or later to be installed.

## Verifying Program Transformations
There are two modes of verification available:

1. One-Shot verification of a single transformation instance (see [xfuzz](#xfuzz))
2. Broad verification of all instances of a given transformation type on a given
   program (see [xfuzzall](#xfuzzall))

### xFUZZ
To verify a single instance of a transformation, make sure you have:

1. A JSON file containing the transformation to be verified
2. An SDFG file with the program the transformation is supposed to operate on.

With both of these things, invoke the verification process using the following
command:

`python -m fuzzyflow.cli.xfuzz -p <PATH_TO_SDFG> -x <PATH_TO_TRANSFORMATION_JSON>`

> Note: For information on how to obtain transformation JSON files, see the
[Good To Know](#good-to-know) section.

A number of command line arguments can further be used to control the verification process:

| argument | Details | Default |
|----------|---------|---------|
| `-r`/`--runs` | Controls the number of trials (i.e., runs with random inputs) | 200 |
| `-c`/`--cutout-strategy` | Control what strategy to use when extracting minimal program cutouts | `SIMPLE` |
| `-s`/`--sampling-strategy` | Control what strategy to use when sampling random inputs | `SIMPLE_UNIFORM` |
| `--data-constraints-file` | Path to a file containing constraints on data values | - |
| `--symbol-constraints-file` | Path to a file containing constraints on symbol values | - |

### xFUZZall
To verify every instance of a transformation on a given program, you need:

1. An SDFG file with the program to check for transformation instances
2. The string descriptor of a transformation. This is a 2-part string separated
   by a `.` (e.g.: `xfclass.xfname`). The first part (`xfclass`) represents what
   class a transformation belongs to (one of `dataflow`, `interstate`,
   `subgraph`, or `passes`), and the second part (`xfname`) represents the class
   name of that transformation (e.g. `GPUTransformMap`).

Run the verification with the following command:

`python -m fuzzyflow.cli.xfuzzall -p <PATH_TO_SDFG> -t <TRANSFORMATION_DESCRIPTOR>`

You can additionally provide a path to an output folder via `-o`/`--output`.
This folder will be created when the verification procedure starts, and will be
used to dump information about any transformation instances that have failed
verification. Specifically, inside this folder, a new folder is created for
each instance of a transformation that fails verification. In each failed case,
the corresponding folder then contains:

1. The SDFG cutout before applying the transformation
2. The SDFG cutout after applying the transformation
3. The transformation instance serialized into a JSON file
4. A file containing the reason the verification failed

If no output path is provided, no information will be stored about instances
that fail verification.

A number of command line arguments can further be used to control the verification process:

| argument | Details | Default |
|----------|---------|---------|
| `-r`/`--runs` | Controls the number of trials (i.e., runs with random inputs) per instance | 200 |
| `-o`/`--output` | Provide a path to an output folder which will be newly created | - |
| `-c`/`--cutout-strategy` | Control what strategy to use when extracting minimal program cutouts | `SIMPLE` |
| `-s`/`--sampling-strategy` | Control what strategy to use when sampling random inputs | `SIMPLE_UNIFORM` |
| `--data-constraints-file` | Path to a file containing constraints on data values | - |
| `--symbol-constraints-file` | Path to a file containing constraints on symbol values | - |
| `--skip-n` | Skip the first N instances of the transformation | - |
| `--savepath` | Keep track of the current progress in a progressfile | - |
| `--restore` | Restore progress previously saved in a progressfile (provide the path via `--savepath`) | - |

## Good To Know

Transformation JSON files can be obtained through the transformation's `.to_json()` method or the DaCe serialization module, or through VSCode:

![image](https://user-images.githubusercontent.com/9193712/180575679-3e5a61c0-9c2d-4332-b377-2f043de9cfa3.png)

