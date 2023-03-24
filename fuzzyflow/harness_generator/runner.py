import numpy as np
import dace
import os
from sdfg2cpp import dump_args, read_args

sdfg_pre = dace.SDFG.from_file('pre.sdfg')
sdfg_post = dace.SDFG.from_file('post.sdfg')

# Inputs / Outputs
I = 4
J = 4
K = 4
flx_field = np.random.rand(I + 1, J, K)
fly_field = np.random.rand(I, J + 1, K)
coeff = np.random.rand(I, J, K)
in_field = np.random.rand(I + 4, J + 4, K)
out_field_pre = np.zeros((I, J, K))     # Output for pre
out_field_post = np.zeros((I, J, K))    # Output for post

# You can test SDFGs in Python
sdfg_pre(flx_field=flx_field, fly_field=fly_field, coeff=coeff, in_field=in_field,
         out_field=out_field_pre, I=I, J=J, K=K)
sdfg_post(flx_field=flx_field, fly_field=fly_field, coeff=coeff, in_field=in_field,
          out_field=out_field_post, I=I, J=J, K=K)

if np.allclose(out_field_pre, out_field_post):
    print('pass')
else:
    print('Pre and Post SDFGs give different results in Python')

# reinitialize, to make sure the harness sees the same data
# This is important, if post writes a 1 where pre left a 0 and you give post_out to C++
# we will miss that bug!
out_field = np.zeros((I, J, K))

# Or create a C++ tester, using the same inputs 
#  - if two SDFGs are given, the outputs are compared in C++
#    (we assume the two SDFGs have the same signature and two different names)
#  - if only one SDFG is given, it is simply called, and the outputs are written back

dump_args("c++", "fuzzer", sdfg_pre, sdfg_post, #target lang, filename, sdfg1, sdfg2
         flx_field=flx_field, fly_field=fly_field, coeff=coeff, in_field=in_field,
         out_field=out_field, I=I, J=J, K=K)

# Build/run the C++ tester
cpp_compiler = "g++"
cpp_flags = "-O3 -march=native"
sdfg1_src = ".dacecache/"+sdfg_pre.name+"/src/cpu/"+sdfg_pre.name+".cpp"
sdfg2_src = ".dacecache/"+sdfg_post.name+"/src/cpu/"+sdfg_post.name+".cpp"
sdfg1_inc = ".dacecache/"+sdfg_pre.name+"/include/"
sdfg2_inc = ".dacecache/"+sdfg_post.name+"/include/"
dace_inc = "../../dace/dace/runtime/include/"
compile_fuzzer_cmd = " ".join([cpp_compiler, cpp_flags, "fuzzer.cpp", sdfg1_src, sdfg2_src, "-I"+sdfg1_inc, "-I"+sdfg2_inc, "-I"+dace_inc])+" -o fuzzer"
print(compile_fuzzer_cmd)
os.system(compile_fuzzer_cmd)
print("./fuzzer fuzzer.dat out1.dat out2.dat")
os.system("./fuzzer fuzzer.dat out1.dat out2.dat")

# Now we read back the data so we can examine in Python 
# You don't need this the C++ tester already compares, this is more for debugging

out_field_pre_c = np.zeros((I, J, K)) #values are not important we just need the memory/size
read_args("out1.dat", 
         flx_field=flx_field, fly_field=fly_field, coeff=coeff, in_field=in_field,
         out_field=out_field_pre_c, I=I, J=J, K=K)
if np.allclose(out_field_pre, out_field_pre_c):
    print('pre python == c++')
else:
    print('pre python != c++')
out_field_post_c = np.zeros((I, J, K)) #values are not important we just need the memory/size
read_args("out2.dat", 
         flx_field=flx_field, fly_field=fly_field, coeff=coeff, in_field=in_field,
         out_field=out_field_post_c, I=I, J=J, K=K)
if np.allclose(out_field_post, out_field_post_c):
    print('post python == c++')
else:
    print('post python != c++')


#
####
# 

