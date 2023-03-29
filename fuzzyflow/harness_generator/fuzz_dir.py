import os
import dace
from dace import dtypes

## PATHS ## (fix those)
DACE_PATH = "/home/timos/Work/dace/"
AFL_PATH = "/home/timos/Work/AFLplusplus/"

def remove_inst(sdfg):
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.sdfg.nodes.AccessNode):
                n.instrument = dtypes.DataInstrumentationType.No_Instrumentation
            else:
                n.instrument = dtypes.InstrumentationType.No_Instrumentation

def fuzz():
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
    flags = "-O3 -fopenmp"

    # first compile using gcc, just make sure it works
    compiler = "g++"
    compile_cmd = " ".join([compiler, flags, "harness.cpp", src_pre, src_post, inc_pre, inc_post, inc_dace, "-o", "harness"])
    print(compile_cmd)
    os.system(compile_cmd)
    os.system("./harness harness.dat out1.dat out2.dat")

    # now lets use afl
    compiler = AFL_PATH+"afl-g++-fast"
    compile_cmd = " ".join([compiler, flags, "harness.cpp", src_pre, src_post, inc_pre, inc_post, inc_dace, "-o", "harness"])
    print(compile_cmd)
    os.system(compile_cmd)
    os.system("rm -rf afl_seeds afl_finds")
    os.system("mkdir afl_seeds")
    os.system("cp harness.dat afl_seeds")
    os.system("mkdir afl_finds")
    afl_cmd = AFL_PATH+"afl-fuzz -i afl_seeds -o afl_finds -t 5000 -V 300 -- ./harness @@ out1.dat out2.dat"
    os.system(afl_cmd)

def traverse_dir(path):
    subdirs = [f.path for f in os.scandir(path) if f.is_dir()]    
    for d in subdirs:
        os.chdir(d)
        fuzz()
        os.chdir("..")

traverse_dir(".")
