import numpy as np
import dace

# my code starts here (need to factor out later)
import struct

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
    elif type(arg) == np.ndarray:
        allocate = True
        elems = arg.size
        if (arg.dtype == np.int32):
            elemsize = 4
            elemtypec = "int"
        elif (arg.dtype == np.float32):
            elemsize = 4
            elemtypec = "float"
        elif (arg.dtype == np.double):
            elemsize = 8
            elemtypec = "double"
        else:
            raise(ValueError("Unsupported np type "+str(arg.dtype)))
    return (allocate, elems, elemsize, elemtypec)


def write_arg(arg, data_file):
    elems = 1
    elemsize = 0
    elemtypec = "void"
    allocate = False
    if isinstance(arg, int):
        elemsize = 4
        data_file.write((arg).to_bytes(elemsize, byteorder='little', signed=True))
    elif type(arg) == np.ndarray:
        allocate = True
        elems = arg.size
        if (arg.dtype == np.int32):
            elemsize = 4
            for x in np.nditer(arg.T):
                data_file.write((x.item()).to_bytes(elemsize, byteorder='little', signed=True))
        elif (arg.dtype == np.float32):
            for x in np.nditer(arg.T):
                ba = bytearray(struct.pack("f", x.item()))  
                data_file.write(ba)
        elif (arg.dtype == np.double):
            for x in np.nditer(arg.T):
                ba = bytearray(struct.pack("d", x.item()))  
                data_file.write(ba)
        else:
            raise(ValueError("Unsupported np type "+str(arg.dtype)))
    else:
        raise ValueError("Unsupported type: "+str(type(arg)))

 
def read_arg(arg, argname, code_file):
    (allocate, elems, elemsize, elemtypec) = get_arg_type(arg, argname)
    # now read the data, lets keep this c compatible
    makeptr=""
    if allocate:
        makeptr="" #this is already a pointer
    else:
        makeptr="&"
    print("  " + "dacefuzz_read_"+str(elemtypec)+"( "+str(makeptr)+str(argname)+", argdata, "+str(elems) +");", file=code_file)


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
    print("    " + "double dacefuzz_diff = (((double)dacefuzz_tmp1) - ((double)dacefuzz_tmp2));", file=code_file)
    print("    " + "if (fabs(dacefuzz_diff) > 0.0) {", file=code_file)
    print("      " + "printf(\"The outputs differ for argument "+str(argname)+" at position %i of %i\\n\", dacefuzz_idx_i, "+str(elems)+"); //exit(EXIT_FAILURE);", file=code_file)
    print("      " + "printf(\"%lf vs %lf\\n\", ((double)dacefuzz_tmp1), ((double)dacefuzz_tmp2));", file=code_file)
    print("    }", file=code_file)
    print("  }", file=code_file)





def print_helpers(code_file):
    print("/* helper functions we use below */", file=code_file)
    print("int dacefuzz_read_void(void* dest, FILE* datastream, size_t elems) { printf(\"Something went wrong during codegen, we could not infer the type of some argument!\"); exit(EXIT_FAILURE); return 0;}", file=code_file)
    print("int dacefuzz_read_int(void* dest, FILE* datastream, size_t elems) { return fread(dest, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_double(void* dest, FILE* datastream, size_t elems) { return fread(dest, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_read_float(void* dest, FILE* datastream, size_t elems) { return fread(dest, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_void(FILE* datastream, void* src, size_t elems) { printf(\"Something went wrong during codegen, we could not infer the type of some argument!\"); exit(EXIT_FAILURE); return 0;}", file=code_file)
    print("int dacefuzz_write_int(FILE* datastream, void* src, size_t elems) { return fwrite(src, 4, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_double(FILE* datastream, void* src, size_t elems) { return fwrite(src, 8, elems, datastream); }", file=code_file)
    print("int dacefuzz_write_float(FILE* datastream, void* src, size_t elems) { return fwrite(src, 4, elems, datastream); }\n", file=code_file)


def generate_call(out_file, sdfg, args, kwargs):
  # we have all the "how to call an sdfg" relevant code in CompiledSDFG, not in SDFG itself. So we approximate it here, yay
  print("  __program_"+sdfg.name+"(", file=out_file, end="")
  print("NULL, ", file=out_file, end="")
  args = sdfg.arglist()
  print(", ".join(args), file=out_file, end="")
  print(");", file=out_file)


def generate_headers(code_file, sdfg1, sdfg2):
    print("#include <cstdio>", file=code_file)
    print("#include <cstdlib>", file=code_file)
    print("#include <cassert>\n", file=code_file)
    print("#include \""+sdfg1.name+".h\"", file=code_file)
    if sdfg2 is not None:
        print("#include \""+sdfg2.name+".h\"", file=code_file)
    print("", file=code_file)
    print("FILE* argdata;", file=code_file)
    print("", file=code_file)

def generate_write_back(code_file, datafile, sdfg, args, kwargs):
    for arg in args: 
        write_back_arg(datafile, arg, None, code_file)
    for arg in kwargs:
        write_back_arg(datafile, kwargs[arg], arg, code_file)


def generate_reads(code_file, sdfg, args, kwargs):
    print("  /* read the input data of "+sdfg.name+" */", file=code_file)
    print("  argdata = fopen(argv[1], \"rb\");", file=code_file);
    print("  if (argdata == NULL) {printf(\"Could not open data file %s!\\n\", argv[1]); exit(EXIT_FAILURE);}\n", file=code_file)
    for arg in args: 
        read_arg(arg, None, code_file)
    for arg in kwargs:
        read_arg(kwargs[arg], arg, code_file)
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

def generate_validators(code_file, allocs, sdfg, args, kwargs):
    print("  /* validate if the input data matches the sdfg argument \"specification\" */", file=code_file)
    arglist = sdfg.arglist()
    for arg in arglist:
        dt = arglist[arg]
        if isinstance(dt, dace.data.Array):
            for alloc in allocs:
                (name, size, elemsize, typec, allocated) = alloc
                if str(name) == str(arg):
                    print("  if ("+str(size)+"*"+str(elemsize) + " != (" + str(dt.total_size) + ")*"+str(elemsize)+") { //check if "+str(arg)+" has correct size (lhs=allocated size, rhs=symbolic size)", file=code_file)
                    print("    printf(\"The size of the passed in "+str(arg)+" ("+str(size)+" elements) does not match its specification in "+sdfg.name+" ("+str(dt.total_size)+")\\n\"); return 0;", file=code_file)
                    # here we could also reallocate to be able to continue - but then the data in the new region is undefined
                    print("  }", file=code_file)
        elif isinstance(dt, dace.data.Scalar):
            pass
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
    print("  fclose(out1);", file=code_file);
    print("  fclose(out2);", file=code_file);
    print("", file=code_file)


def dump_args(out_lang, out_file, sdfg1, sdfg2, *args, **kwargs):
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
        generate_reads(code_file, sdfg1, args, kwargs) # read the inputs
        generate_validators(code_file, allocs, sdfg1, args, kwargs)
        print("", file=code_file)
        print("  /* call "+ sdfg1.name + " */", file=code_file)
        generate_call(code_file, sdfg1, args, kwargs) # generate the call
        print("", file=code_file)
        print("  FILE* outdata1 = fopen(argv[2], \"wb\");", file=code_file);
        generate_write_back(code_file, "outdata1", sdfg1, args, kwargs) # write back the data in c++
        print("  fclose(outdata1);\n", file=code_file)
        if sdfg2 is not None:
            generate_reads(code_file, sdfg2, args, kwargs) # read the inputs
            generate_validators(code_file, allocs, sdfg2, args, kwargs)
            print("", file=code_file)
            print("  /* call "+ sdfg2.name + " */", file=code_file)
            generate_call(code_file, sdfg2, args, kwargs) # generate the call
            print("", file=code_file)
            print("  /* write all data of "+sdfg2.name+" back into an output file */", file=code_file)
            print("  FILE* outdata2 = fopen(argv[3], \"wb\");", file=code_file);
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
        for arg in args:  
            write_arg(arg, data_file)
        for arg in kwargs:
            write_arg(kwargs[arg], data_file)


