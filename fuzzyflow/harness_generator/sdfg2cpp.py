import numpy as np
import dace
from dace.symbolic import symstr

import struct

defined_types = [
        {'c_type': 'int',             'fmtstring': '%i',       'pack':'i', 'is_complex':False,  'size': 4, 'nptype': np.int32},
        {'c_type': 'double',          'fmtstring': '%lf',      'pack':'d',  'is_complex':False, 'size': 8, 'nptype': np.double},
        {'c_type': 'float',          'fmtstring': '%f',      'pack':'f',  'is_complex':False, 'size': 4, 'nptype': np.float32},
        {'c_type': 'DACE_INT64',      'fmtstring': '%lli',     'pack':'q',  'is_complex':False, 'size': 8, 'nptype': np.int64},
        {'c_type': 'DACE_UINT64',     'fmtstring': '%llu',     'pack':'Q',  'is_complex':False, 'size': 8, 'nptype': np.uint64},
        {'c_type': 'DACE_UINT8',     'fmtstring': '%u',     'pack':'c',  'is_complex':False, 'size': 1, 'nptype': np.uint8},
        {'c_type': 'DACE_COMPLEX128', 'fmtstring': '%lf+%lfj', 'pack':'dd',  'is_complex':True, 'size': 16, 'nptype': np.complex128},
]

def type_conv(frm, to, val):
    for t in defined_types:
        if t[frm] == val:
            return t[to]
    raise ValueError("Type conversion failed: ("+frm+" -> "+to+" for "+str(val))


def get_arg_type(arg, argname):
    elems = 1
    elemsize = 0
    elemtypec = "void"
    allocate = False
    if argname is None:
        argname = "pos_arg"
    if isinstance(arg, int):
        elemtypec = "int"
        elemsize = 4
    elif isinstance(arg, float):
        elemtypec = "double"
        elemsize = 8
    elif type(arg) == np.ndarray:
        allocate = True
        elems = arg.size
        elemsize = type_conv('nptype', 'size', arg.dtype)
        elemtypec = type_conv('nptype', 'c_type', arg.dtype)
    return (allocate, elems, elemsize, elemtypec)


def write_arg(arg, data_file):
    elems = 1
    elemsize = 0
    elemtypec = "void"
    allocate = False
    if isinstance(arg, int):
        elemsize = 4
        data_file.write((arg).to_bytes(elemsize, byteorder='little', signed=True))
    elif isinstance(arg, float):
        ba = bytearray(struct.pack("f", arg))
        data_file.write(ba)
    elif type(arg) == np.ndarray:
        elems = arg.size
        packstr = type_conv('nptype', 'pack', arg.dtype)
        is_complex = type_conv('nptype', 'is_complex', arg.dtype)
        for x in np.nditer(arg.T):
            if is_complex:
                ba = bytearray(struct.pack(packstr, x.item().real, xitem().imag))
            else:
                ba = bytearray(struct.pack(packstr, x.item()))
            data_file.write(ba)
    else:
        raise ValueError("Unsupported type: "+str(type(arg)))



def read_arg(arg, argname, sdfg, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    # now read the data, lets keep this c compatible
    makeptr=""
    if allocate:
        makeptr="" #this is already a pointer
    else:
        makeptr="&"
    print("  " + "dacefuzz_read_"+str(elemtypec)+"( "+str(makeptr)+str(argname)+", argdata, "+str(elems) +");", file=code_file)

def print_arg(arg, argname, sdfg, prefix, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    args = sdfg.arglist()
    dt = args[argname]
    fmt_string = type_conv(frm='c_type', to='fmtstring', val=elemtypec)
    if not allocate:
        print("  printf(\""+prefix+"%s = "+fmt_string+"\\n\", \""+str(argname)+"\", "+str(argname) +");", file=code_file) #give some insight into symbols for debugging
    else:
        print("  printf(\""+prefix+"%s = np.ndarray(shape="+str(dt.shape)+", order=\\\"C\\\", dtype="+str(dt.dtype)+", buffer=np.array([\",\""+str(argname)+"\");", file=code_file)
        print("  for (size_t i=0; i<"+symstr(dt.total_size)+"; i++) {", file=code_file)
        if type_conv('c_type', 'is_complex', elemtypec):
            print("    printf(\""+fmt_string+" , \", "+str(argname)+"[i].real(), "+str(argname)+"[i].imag());", file=code_file)
        else:
            print("    printf(\""+fmt_string+" , \", "+str(argname)+"[i]);", file=code_file)
        print("    if (i%10==0) printf(\"\\n\");", file=code_file)
        print("  }", file=code_file)
        print("  printf(\"]))\\n\");", file=code_file)


def generate_repro_py(code_file, sdfg, prefix, args, kwargs):
    arglist = sdfg.arglist()
    # print scalars first, since array shapes might depend on them
    for arg in arglist:
        if isinstance(arglist[arg], dace.data.Scalar):
            print_arg(kwargs[arg], arg, sdfg, prefix, code_file)
    # now print arrays and ignore the scalars
    for arg in arglist:
        if isinstance(arglist[arg], dace.data.Scalar):
            pass
        elif isinstance(arglist[arg], dace.data.Array):
            print_arg(kwargs[arg], arg, sdfg, prefix, code_file)
        else:
            raise ValueError("Datatype not implemented!")


def autogen_arg(arg, argname, initializer, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    makeptr=""
    if allocate:
        makeptr="" #this is already a pointer
    else:
        makeptr="&"
    print("  " + "dacefuzz_init_"+str(initializer)+"_"+str(elemtypec)+"( "+str(makeptr)+str(argname)+", "+str(elems) +");", file=code_file)


def alloc_arg(arg, argname, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    if allocate:
        print("  " + str(elemtypec)+"* " + str(argname)+" = ("+str(elemtypec)+"*) malloc("+str(elems)+"*"+str(elemsize)+");", file=code_file)
        print("  " + "assert("+str(argname)+" != NULL); //check if allocation was successful", file=code_file)
        return [ (argname, elems, elemsize, elemtypec, True) ]
    else:
        print("  "+elemtypec+" "+argname+";", file=code_file)
        return [ (argname, elems, elemsize, elemtypec, False) ]


def write_back_arg(outfile_name, arg, argname, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    if allocate:
        print("  dacefuzz_write_"+elemtypec+"("+outfile_name+", "+argname+", "+str(elems)+");", file=code_file)
    else:
        print("  dacefuzz_write_"+elemtypec+"("+outfile_name+", &"+argname+", "+str(elems)+");", file=code_file)


def compare_arg(arg, argname, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    print("  for (int dacefuzz_idx_i=0; dacefuzz_idx_i<"+str(elems)+"; dacefuzz_idx_i++) {", file=code_file)
    print("    "+str(elemtypec) +" dacefuzz_tmp1;", file=code_file)
    print("    "+str(elemtypec) +" dacefuzz_tmp2;", file=code_file)
    print("    " + "dacefuzz_read_"+str(elemtypec)+"(&dacefuzz_tmp1, out1, 1);", file=code_file)
    print("    " + "dacefuzz_read_"+str(elemtypec)+"(&dacefuzz_tmp2, out2, 1);", file=code_file)
    if type_conv('c_type', 'is_complex', elemtypec):
        print("    " + "double dacefuzz_diff_real = fabs(((double)dacefuzz_tmp1.real()) - ((double)dacefuzz_tmp2.real()));", file=code_file)
        print("    " + "double dacefuzz_diff_imag = fabs(((double)dacefuzz_tmp1.imag()) - ((double)dacefuzz_tmp2.imag()));", file=code_file)
        print("    " + "double dacefuzz_diff = dacefuzz_diff_real + dacefuzz_diff_imag;", file=code_file)
    else:
        print("    " + "double dacefuzz_diff = (((double)dacefuzz_tmp1) - ((double)dacefuzz_tmp2));", file=code_file)
    print("    " + "if (fabs(dacefuzz_diff) > 0.0001) {", file=code_file)
    print("      " + "printf(\"The outputs differ for argument "+str(argname)+" at position %i of %i by %lf\\n\", dacefuzz_idx_i, "+str(elems)+", dacefuzz_diff);", file=code_file)
    print("      " + "*((int*) 0) = 0; //make afl happy", file=code_file)
    print("      " + "exit(EXIT_FAILURE);", file=code_file)
    print("    }", file=code_file)
    print("  }", file=code_file)


def print_helpers(code_file):
    print("/* helper functions we use below */", file=code_file)
    print("int dacefuzz_read_void(void* dest, FILE* datastream, size_t elems) { printf(\"Something went wrong during codegen, we could not infer the type of some argument!\"); exit(EXIT_FAILURE); return 0;}", file=code_file)
    print("int dacefuzz_read_int(void* dest, FILE* datastream, size_t elems) { return fread(dest, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_DACE_INT64(void* dest, FILE* datastream, size_t elems) { return fread(dest, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_DACE_UINT64(void* dest, FILE* datastream, size_t elems) { return fread(dest, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_DACE_UINT8(void* dest, FILE* datastream, size_t elems) { return fread(dest, 1, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_double(void* dest, FILE* datastream, size_t elems) { return fread(dest, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_DACE_COMPLEX128(void* dest, FILE* datastream, size_t elems) {int ret=0; for (size_t i=0; i<elems; i++) {double t[2];  ret+=fread(t, 16, 1, datastream); ((dace::complex128*)(dest))[i] = dace::complex128(t[0], t[1]); } return ret; }", file=code_file)
    print("int dacefuzz_read_float(void* dest, FILE* datastream, size_t elems) { return fread(dest, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_void(FILE* datastream, void* src, size_t elems) { printf(\"Something went wrong during codegen, we could not infer the type of some argument!\"); exit(EXIT_FAILURE); return 0;}", file=code_file)
    print("int dacefuzz_write_int(FILE* datastream, void* src, size_t elems) { return fwrite(src, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_DACE_INT64(FILE* datastream, void* src, size_t elems) { return fwrite(src, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_DACE_UINT64(FILE* datastream, void* src, size_t elems) { return fwrite(src, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_DACE_UINT8(FILE* datastream, void* src, size_t elems) { return fwrite(src, 1, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_double(FILE* datastream, void* src, size_t elems) { return fwrite(src, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_float(FILE* datastream, void* src, size_t elems) { return fwrite(src, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_DACE_COMPLEX128(FILE* datastream, void* src, size_t elems) {int ret=0; for (size_t i=0; i<elems; i++) {double t[2]; t[0]=((dace::complex128*)src)[i].real(); t[1]=((dace::complex128*)(src))[i].imag(); ret += fwrite(t,16,1,datastream); } return ret; }", file=code_file)
    print("int dacefuzz_init_zeros_int(int* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=0; } return elems; }", file=code_file)
    print("int dacefuzz_init_zeros_DACE_INT64(DACE_INT64* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=0; } return elems; }", file=code_file)
    print("int dacefuzz_init_zeros_DACE_UINT64(DACE_UINT64* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=0; } return elems; }", file=code_file)
    print("int dacefuzz_init_zeros_DACE_UINT8(DACE_UINT64* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=0; } return elems; }", file=code_file)
    print("int dacefuzz_init_zeros_float(float* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=0.0; } return elems; }", file=code_file)
    print("int dacefuzz_init_zeros_double(double* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=0.0; } return elems; }", file=code_file)
    print("int dacefuzz_init_zeros_DACE_COMPLEX128(DACE_COMPLEX128* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i] = dace::complex128(0.0, 0.0); } return elems; }\n", file=code_file)
    print("int dacefuzz_init_rand_int(int* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=rand(); } return elems; }", file=code_file)
    print("int dacefuzz_init_rand_DACE_INT64(DACE_INT64 * dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=rand(); } return elems; }", file=code_file)
    print("int dacefuzz_init_rand_DACE_UINT64(DACE_UINT64 * dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]=rand(); } return elems; }", file=code_file)
    print("int dacefuzz_init_rand_DACE_UINT8(DACE_UINT8 * dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]= (rand() % 256); } return elems; }", file=code_file)
    print("int dacefuzz_init_rand_float(float* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]= ((float)rand() / (float) RAND_MAX)*42.0; } return elems; }", file=code_file)
    print("int dacefuzz_init_rand_double(double* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i]= ((double)rand() / (double) RAND_MAX)*42.0; } return elems; }\n", file=code_file)
    print("int dacefuzz_init_rand_DACE_COMPLEX128(DACE_COMPLEX128* dst, size_t elems) { for (size_t i=0; i<elems; i++) { dst[i] = dace::complex128((double) rand() / RAND_MAX, (double) rand() / RAND_MAX); } return elems; }\n", file=code_file)



def generate_call(out_file, sdfg, args, kwargs):
  print("  "+sdfg.name+"Handle_t "+sdfg.name+"_handle = __dace_init_"+sdfg.name+"("+sdfg.init_signature(for_call=True)+");", file=out_file)
  # we have all the "how to call an sdfg" relevant code in CompiledSDFG, not in SDFG itself. So we approximate it here, yay
  print("  __program_"+sdfg.name+"(", file=out_file, end="")
  args = [sdfg.name+"_handle"] + list(sdfg.arglist())
  print(", ".join(args), file=out_file, end="")
  print(");", file=out_file)
  print("  __dace_exit_"+sdfg.name+"("+sdfg.name+"_handle);", file=out_file)
  


def generate_headers(code_file, sdfg1, sdfg2):
    print("#include <cstdio>", file=code_file)
    print("#include <cstdlib>", file=code_file)
    print("#include <cassert>\n", file=code_file)
    print("#include \""+sdfg1.name+".h\"", file=code_file)
    if sdfg2 is not None:
        print("#include \""+sdfg2.name+".h\"", file=code_file)
    print("", file=code_file)
    print("// DaCe make some assumptions on datatype sizes during codegen that might not be true - this is how we conform to its imaginary world", file=code_file)
    print("#define DACE_INT64 long long int", file=code_file)
    print("#define DACE_UINT64 long long unsigned int", file=code_file)
    print("#define DACE_INT8 char", file=code_file)
    print("#define DACE_UINT8 unsigned char", file=code_file)
    print("#define DACE_COMPLEX128 dace::complex128", file=code_file)
    print("", file=code_file)
    print("FILE* argdata;", file=code_file)
    print("int dacefuzz_seed;", file=code_file)
    print("", file=code_file)

def generate_write_back(code_file, datafile, sdfg, args, kwargs):
    for arg in args:
        write_back_arg(datafile, arg, None, code_file)
    for arg in kwargs:
        write_back_arg(datafile, kwargs[arg], arg, code_file)


def generate_reads(code_file, autoinit_args, sdfg, args, kwargs):
    print("  /* read the input data of "+sdfg.name+" */", file=code_file)
    print("  argdata = fopen(argv[1], \"rb\");", file=code_file);
    print("  if (argdata == NULL) {printf(\"Could not open data file %s!\\n\", argv[1]); exit(EXIT_FAILURE);}\n", file=code_file)
    print("  dacefuzz_read_int(&dacefuzz_seed, argdata, 1);", file=code_file)
    print("  printf(\"Random seed used for autoinit args: %i\\n\", dacefuzz_seed);", file=code_file);
    print("  srand(dacefuzz_seed);", file=code_file)
    for arg in args:
        read_arg(arg, None, sdfg, code_file)
    for arg in kwargs:
        if str(arg) in autoinit_args.keys():
            autogen_arg(kwargs[arg], arg, autoinit_args[str(arg)], code_file)
        else:
            read_arg(kwargs[arg], arg, sdfg, code_file)
    print("  fclose(argdata);", file=code_file)


def generate_allocs(code_file, sdfg, args, kwargs):
    print("  /* allocate in/out data */", file=code_file)
    allocated_syms = []
    for arg in args:
        allocated_syms += alloc_arg(arg, None, code_file)
    for arg in kwargs:
        allocated_syms += alloc_arg(kwargs[arg], arg, code_file)
    print("", file=code_file)
    return allocated_syms


def generate_validators(code_file, allocs, sym_constraints, autoinit_args, sdfg, args, kwargs):
    print("  /* validate if the input data matches the sdfg argument \"specification\" */", file=code_file)
    arglist = sdfg.arglist()
    for arg in arglist:
        dt = arglist[arg]
        if isinstance(dt, dace.data.Array):
            pass # we validate those below, scalars first, as they determine array sizes
        elif isinstance(dt, dace.data.Scalar):
            if arg in sym_constraints:
                print('  if ('+str(arg) + '<' + symstr(sym_constraints[arg][0]) + '){printf(\"Current symbol doesn\'t fit lower constraint - bail out.\\n\"); return 0;}' , file=code_file)
                print('  if ('+str(arg) + '>' + symstr(sym_constraints[arg][1]) + '){printf(\"Current symbol doesn\'t fit upper constraint - bail out.\\n\"); return 0;}' , file=code_file)
        elif arg in sym_constraints:
            print("Hitting a part of untested code! Now you found a test :)")
            exit(1)
            print('  if ('+str(arg) + '<' + symstr(sym_constraints[arg][0]) + '){printf(\"Current symbol doesn\'t fit lower constraint - bail out.\\n\"); return 0;}' , file=code_file)
            print('  if ('+str(arg) + '>' + symstr(sym_constraints[arg][1]) + '){printf(\"Current symbol doesn\'t fit upper constraint - bail out.\\n\"); return 0;}' , file=code_file)
        else:
            raise ValueError("Unsupported data type found while generating validators")
    for arg in arglist:
        dt = arglist[arg]
        if isinstance(dt, dace.data.Array):
            for alloc in allocs:
                (name, size, elemsize, typec, allocated) = alloc
                if str(name) == str(arg):
                    print("  if ("+str(size)+"*"+str(elemsize) + " < (" + symstr(dt.total_size) + ")*"+str(elemsize)+") { //check if "+str(arg)+" has correct size (lhs=allocated size, rhs=symbolic size)", file=code_file)
                    print("    printf(\"The size of the passed in "+str(arg)+" ("+str(size)+" elements) does not match its specification in "+sdfg.name+" ("+symstr(dt.total_size)+"=%lf MB) - resizing\\n\", (double)("+symstr(dt.total_size)+"*"+str(elemsize)+")/1000000.0);", file=code_file)
                    print("    "+str(name)+ " = ("+str(typec)+"*) realloc("+str(name)+", (" + symstr(dt.total_size) + ")*"+str(elemsize)+");", file=code_file)
                    print("    assert("+str(name)+" != NULL);", file=code_file)
                    if arg in autoinit_args.keys():
                        print("  " + "dacefuzz_init_"+str(autoinit_args[arg])+"_"+str(typec)+"("+str(arg)+", "+symstr(dt.total_size) +");", file=code_file)
                    print("  }", file=code_file)
        elif isinstance(dt, dace.data.Scalar):
            pass
        elif arg in sym_constraints:
            print("Hitting a part of untested code! Now you found a test :)")
            exit(1)
            print('  if ('+str(arg) + '<' + symstr(sym_constraints[arg][0]) + '){printf(\"Current symbol doesn\'t fit lower constraint - bail out.\\n\"); return 0;}' , file=code_file)
            print('  if ('+str(arg) + '>' + symstr(sym_constraints[arg][1]) + '){printf(\"Current symbol doesn\'t fit upper constraint - bail out.\\n\"); return 0;}' , file=code_file)
        else:
            raise ValueError("Unsupported data type found while generating validators")


def compare_outputs(code_file, sdfg, args, kwargs):
    print("  /* compare the outputs of both sdfgs */", file=code_file)
    print("  FILE* out1 = fopen(argv[2], \"rb\");", file=code_file)
    print("  FILE* out2 = fopen(argv[3], \"rb\");", file=code_file)
    for arg in args:
        compare_arg(arg, None, code_file)
    for arg in kwargs:
        compare_arg(kwargs[arg], arg, code_file)
    print("  fclose(out1);", file=code_file)
    print("  fclose(out2);", file=code_file)
    print("", file=code_file)


def dump_args(out_lang, out_file, autoinit_args, sym_constraints, sdfg1, sdfg2, *args, **kwargs):
    if out_lang == "c++":
        code_ext = ".cpp"
    if out_lang != "c++":
        raise ValueError("Unsupported output language!")

    with open(out_file + code_ext, "w") as code_file:
        # write out code that opens data file and reads it, calls sdfg
        generate_headers(code_file, sdfg1, sdfg2)
        print_helpers(code_file)
        print("int main(int argc, char** argv) {", file=code_file)
        if sdfg2 is None:
            print("  if (argc < 3) {\n    printf(\"Call this verifier with a data in and out argument, i.e: %s data_in data_out\\n\", argv[0]); exit(EXIT_FAILURE);\n  }\n", file=code_file)
        else:
            print("  if (argc < 4) {\n    printf(\"Call this verifier with a data in and two datafile arguments, i.e: %s data_in data_out_1 data_out_2\\n\", argv[0]); exit(EXIT_FAILURE);\n  }\n", file=code_file)
        allocs = generate_allocs(code_file, sdfg1, args, kwargs) # allocate mem
        generate_reads(code_file, autoinit_args, sdfg1, args, kwargs) # read the inputs
        sizes = generate_validators(code_file, allocs, sym_constraints, autoinit_args, sdfg1, args, kwargs)
        generate_repro_py(code_file, sdfg1, "sdfg1_", args, kwargs)
        print("", file=code_file)
        print("  /* call "+ sdfg1.name + " */", file=code_file)
        generate_call(code_file, sdfg1, args, kwargs) # generate the call
        print("", file=code_file)
        print("  FILE* outdata1 = fopen(argv[2], \"wb\");", file=code_file)
        generate_write_back(code_file, "outdata1", sdfg1, args, kwargs) # write back the data in c++
        print("  fclose(outdata1);\n", file=code_file)
        if sdfg2 is not None:
            generate_reads(code_file, autoinit_args, sdfg2, args, kwargs) # read the inputs
            sizes = generate_validators(code_file, allocs, sym_constraints, autoinit_args, sdfg2, args, kwargs)
            generate_repro_py(code_file, sdfg2, "sdfg2_", args, kwargs)
            print("", file=code_file)
            print("  /* call "+ sdfg2.name + " */", file=code_file)
            generate_call(code_file, sdfg2, args, kwargs) # generate the call
            print("", file=code_file)
            print("  /* write all data of "+sdfg2.name+" back into an output file */", file=code_file)
            print("  FILE* outdata2 = fopen(argv[3], \"wb\");", file=code_file)
            generate_write_back(code_file, "outdata2", sdfg2, args, kwargs) # write back the data in c++
            print("  fclose(outdata2);\n", file=code_file)
            compare_outputs(code_file, sdfg2, args, kwargs) # compare the outputs if we have two sdfgs

        print("  /* free input data buffers */", file=code_file)
        for s in allocs:
            (argname, elems, elemsize, typec, allocate) = s
            if allocate:
                print("  free("+str(argname)+");", file=code_file)
        print("}", file=code_file)

    with open(out_file + ".dat", "wb") as data_file:
        # serialize arguments as binaries
        data_file.write(int(23).to_bytes(4, byteorder='little', signed=True)) #write a random seed - will be modified by fuzzer
        for arg in args:
            write_arg(arg, data_file)
        for arg in kwargs:
            if str(arg) in autoinit_args.keys():
                print(str(arg) + " is an autoinit arg, it will NOT be present in the generated input file!")
            else:
                write_arg(kwargs[arg], data_file)


def read_back_arg(arg, argname, data_file):
    elems = 1
    elemsize = 0
    if isinstance(arg, int):
        elemsize = 4
        data_file.read(elemsize)
        # we can't write back an integer (passed by value)
    elif isinstance(arg, float):
        elemsize = 8
        data_file.read(elemsize)
        # we can't write back an float (passed by value, stored as double)
    elif type(arg) == np.ndarray:
        elems = arg.size
        for i in range(0, elems):
            elemsize = type_conv('nptype', 'size', arg.dtype)
            packstr = type_conv('nptype', 'pack', arg.dtype)
            data = data_file.read(elemsize)
            value = struct.unpack('i', data)[0]
            arg.flat[i] = value
    else:
        raise ValueError("Unsupported type: "+str(type(arg)))


def read_args(datafile, *args, **kwargs): #this reads the datafile the harness produces
    with open(datafile, "rb") as data_file:
        for arg in args:
            read_back_arg(arg, None, data_file)
        for arg in kwargs:
            read_back_arg(kwargs[arg], arg, data_file)
