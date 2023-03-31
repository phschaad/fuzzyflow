#!/usr/bin/python3

import os
import sys
import glob
import dace
from dace import dtypes

## PATHS ## (fix those)
DACE_PATH = "/home/timos/Work/dace/"
AFL_PATH = "/home/timos/Work/AFLplusplus/"
sdfg2cpp_path = "/home/timos/Work/fuzzyflow/fuzzyflow/harness_generator/sdfg2cpp.py"
depickler_path = "/home/timos/Work/fuzzyflow/fuzzyflow/harness_generator/depickle.py"

def remove_inst(sdfg):
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.sdfg.nodes.AccessNode):
                n.instrument = dtypes.DataInstrumentationType.No_Instrumentation
            else:
                n.instrument = dtypes.InstrumentationType.No_Instrumentation

def harness_regen():
    os.system("cp "+sdfg2cpp_path+" .")
    os.system("cp "+depickler_path+" .")
    os.system("python3 ./depickle.py")

def fuzz():
    print("Preparing fuzzer in "+os.getcwd())
    sdfg_pre = dace.SDFG.from_file('pre.sdfg')
    sdfg_post = dace.SDFG.from_file('post.sdfg')
    remove_inst(sdfg_pre)
    remove_inst(sdfg_post)
    sdfg_pre.compile()
    sdfg_post.compile()

    src_pre = ".dacecache/"+sdfg_pre.name+"/src/cpu/"+sdfg_pre.name+".cpp"
    src_post = ".dacecache/"+sdfg_post.name+"/src/cpu/"+sdfg_post.name+".cpp"
    inc_pre = "-I.dacecache/"+sdfg_pre.name+"/include/"
    inc_post = "-I.dacecache/"+sdfg_post.name+"/include/"
    inc_dace = "-I"+DACE_PATH+"dace/runtime/include/"
    inc_cuda = "-I/usr/local/cuda/include"
    libs_cuda = "-L/usr/local/cuda/lib64/ -lcudart -L.dacecache/"+sdfg_post.name+"/build/ -l"+sdfg_post.name+" -Wl,-rpath="+os.getcwd()+"/.dacecache/"+sdfg_post.name+"/build/"
    flags = "-O3 -fopenmp -DWITH_CUDA"

    # first compile using gcc, just make sure it works
    compiler = "g++"
    compile_cmd = " ".join([compiler, flags, "harness.cpp", src_pre, src_post, inc_pre, inc_post, inc_dace, inc_cuda, libs_cuda, "-o", "harness"])
    print(compile_cmd)
    ret  = os.system(compile_cmd)
    if ret != 0:
        os.exit()
    os.system("./harness harness.dat out1.dat out2.dat")

    # now lets use afl
    compiler = AFL_PATH+"afl-g++-fast"
    compile_cmd = " ".join([compiler, flags, "harness.cpp", src_pre, src_post, inc_pre, inc_post, inc_dace, inc_cuda, libs_cuda, "-o", "harness"])
    print(compile_cmd)
    os.system(compile_cmd)
    os.system("rm -rf afl_seeds afl_finds")
    os.system("mkdir afl_seeds")
    os.system("cp harness.dat afl_seeds")
    os.system("mkdir afl_finds")
    afl_cmd = AFL_PATH+"afl-fuzz -i afl_seeds -o afl_finds -t 10000 -V 10 -- ./harness @@ out1.dat out2.dat"
    os.system(afl_cmd)

def traverse_dir(path):
    fuzzdirs = []
    for root, dirs, files in os.walk(path):
        for dirname in dirs:
            newpath = (os.path.join(root, dirname))
            if os.path.isfile(os.path.join(os.getcwd(), newpath, "harness.cpp")):
                fuzzdirs += [newpath]
    
    for f in fuzzdirs:
        print("Will fuzz in "+f)
        origwd = os.getcwd()
        os.chdir(f)
        harness_regen()
        fuzz()
        os.chdir(origwd)

if len(sys.argv) > 1:
    traverse_dir(sys.argv[1])
else:
    traverse_dir(".")
