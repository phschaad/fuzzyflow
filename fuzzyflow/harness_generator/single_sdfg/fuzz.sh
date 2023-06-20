rm -rf afl_out/
python3 ./runner.py
/home/timos/Work/fuzzyflow/fuzzyflow/afl++/AFLplusplus-4.05c/afl-g++-fast  -O3 -march=native fuzzer.cpp .dacecache/hdiff_cutout_pre/src/cpu/hdiff_cutout_pre.cpp .dacecache/hdiff_cutout_post/src/cpu/hdiff_cutout_post.cpp -I.dacecache/hdiff_cutout_pre/include/ -I.dacecache/hdiff_cutout_post/include/ -I../../dace/dace/runtime/include/ -o fuzzer
/home/timos/Work/fuzzyflow/fuzzyflow/afl++/AFLplusplus-4.05c/afl-fuzz -i afl_seeds -o afl_out ./fuzzer @@ out1.dat out2.dat
