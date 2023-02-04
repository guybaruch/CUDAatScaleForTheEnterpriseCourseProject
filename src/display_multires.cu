
#include <stdio.h>
#include <stdlib.h>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string.h>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>


struct Args {
    int k;
    char* input_name;
    char* output_name;
    int num_channels;
};

Args parse_args(int argc, char** argv) {
    if (argc!=5) {
        std::cerr << "Format:\n display_multires "
            "<NUM_RES> <INPUT_FILE> <OUTPUT_FILE> <NUMCHANNELS>\n";
        exit(-1);
    }
    Args args;
    //for (int ci=0; ci<argc; ci++) 
    //    std::cout << "arg " << ci << " " << std::string(argv[ci]) << std::endl;
    {
        args.k = atoi(argv[1]);
        assert(args.k>0 && args.k<7);
    }
    args.input_name = argv[2];
    args.output_name = argv[3];
    {
        args.num_channels = atoi(argv[4]);
        assert(args.num_channels==1 || args.num_channels==3);
    }
    return args;
}
    

int main (int argc, char **argv)
{

   Args args = parse_args(argc, argv); 


   exit (0);
}

