#!/usr/bin/python3

import os
import sys
import glob
import dace
import subprocess
from dace import dtypes

DACE_PATH = "/app/dace/"
sdfg2cpp_path = "/app/fuzzyflow/fuzzyflow/harness_generator/sdfg2cpp.py"
depickler_path = "/app/fuzzyflow/fuzzyflow/harness_generator/depickle.py"

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
    ret = os.system("python3 ./depickle.py")
    assert(ret == 0)

def fuzz(compile_only=False):
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

    if compile_only:
        pass
    else:
        # now lets use afl
        compiler = "afl-g++-fast"
        compile_cmd = " ".join([compiler, flags, "harness.cpp", src_pre, src_post, inc_pre, inc_post, inc_dace, inc_cuda, libs_cuda, "-o", "harness"])
        print(compile_cmd)
        os.system(compile_cmd)
        os.system("rm -rf afl_seeds afl_finds")
        os.system("mkdir afl_seeds")
        os.system("cp harness.dat afl_seeds")
        os.system("mkdir afl_finds")
        tasks = []
        afl_cmd = "afl-fuzz -i afl_seeds -o afl_finds -t 10000 -V 60 -M fuzzer0 -- ./harness @@ out1.dat out2.dat"
        p = subprocess.Popen(afl_cmd, shell=True)
        tasks += [p]
        #for t in range(1, 7):
        #    afl_cmd = AFL_PATH+"afl-fuzz -i afl_seeds -o afl_finds -t 10000 -V 60 -S fuzzer"+str(t)+"-- ./harness @@ out1.dat out2.dat"
        #    p = subprocess.Popen(afl_cmd, stdout=subprocess.DEVNULL, shell=True)
        #    tasks += [p]
        for t in tasks:
            t.wait()



def traverse_dir(path):
    fuzzdirs = []
    for root, dirs, files in os.walk(path):
        for dirname in dirs:
            newpath = (os.path.join(root, dirname))
            if os.path.isfile(os.path.join(os.getcwd(), newpath, "harness.cpp")):
                fuzzdirs += [newpath]

    if fuzzdirs == []:
        fuzzdirs = [path]

    for f in fuzzdirs:
        print("Will fuzz in "+f)
        origwd = os.getcwd()
        os.chdir(f)
        harness_regen()
        fuzz(compile_only=False)
        os.chdir(origwd)

if len(sys.argv) > 1:
    traverse_dir(sys.argv[1])
else:
    traverse_dir(".")
