
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

#define CUDA_CALL( call ) \
{\
     auto err = call;\
     if (cudaSuccess !=err) {\
         printf("error %d %s in line %d", err, cudaGetErrorName(err), __LINE__); \
         exit(-1); \
     } \
}

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}


__host__ __device__ struct Rectangle {
    int x0;
    int y0;
    int width;
    int stride;
    int height;
    
    __host__ __device__ Rectangle(int x0, int y0, int width, int height, int pitch):
        x0(x0), y0(y0), width(width), height(height), stride(pitch) {}
    __host__ __device__ inline int x_end() {return x0 + width;}
    __host__ __device__ inline int x_stride_end() {return x0 + stride;}
    __host__ __device__ inline int y_end() {return y0 + height;}
    __host__ __device__ inline int numElements() {return height*stride;}
};


struct Args {
    int k;
    char* input_name;
    char* output_name;
    int num_channels;
};

Args parse_args(int argc, char** argv) {
    Args args;
    if (1) {
        args.k = 1;
        args.input_name = "data/sloth-gray.png";
        args.output_name = "data/sloth-gray-mr.png";
        return args;
    }

    if (argc!=4) {
        std::cerr << "Format:\n display_multires "
            "<NUM_RES> <INPUT_FILE> <OUTPUT_FILE>\n";
        exit(-1);
    }
    //for (int ci=0; ci<argc; ci++) 
    //    std::cout << "arg " << ci << " " << std::string(argv[ci]) << std::endl;
    {
        args.k = atoi(argv[1]);
        assert(args.k>0 && args.k<7);
    }
    args.input_name = argv[2];
    args.output_name = argv[3];
    return args;
}


#define X_STRIDE 64

// copy src image to first 2/3rds of image, fill the rest with zeros.
__global__ void init_trg_kernel(const Npp8u* src, Npp8u* trg,
        Rectangle srcRect, Rectangle trgRect) {
    const int tidx_x = gridDim.x * blockIdx.x + threadIdx.x;
    const int tidx_y = gridDim.y * blockIdx.y + threadIdx.y;

    const int src_addr = tidx_y * srcRect.stride + tidx_x;
    const int trg_addr = tidx_y * trgRect.stride + tidx_x;

    __shared__ Npp8u buff[X_STRIDE];

    if (tidx_x < srcRect.width && tidx_y < srcRect.height
            && src_addr < srcRect.numElements()) {
        buff[threadIdx.x] = src[src_addr];
    } else {
        buff[threadIdx.x] = 0;
    }
    __syncthreads();

    if (tidx_x < trgRect.stride && tidx_y < trgRect.height
            && trg_addr < trgRect.numElements()) {
        trg[trg_addr] = buff[threadIdx.x];
    }
}

int main (int argc, char **argv)
{

    cudaDeviceReset();

    Args args = parse_args(argc, argv); 
    
    //findCudaDevice(argc, (const char **)argv);
    //gpuDeviceInit(0);
  
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
             libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
            (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
            (runtimeVersion % 100) / 10);

    checkCudaErrors(cudaSetDevice(0));

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 h_src_C1;

    // load gray-scale image from disk, and create device image object.
    npp::loadImage(args.input_name, h_src_C1);
    npp::ImageNPP_8u_C1 d_src_C1(h_src_C1);
    
    Rectangle src_rect(0, 0, (int)d_src_C1.width(), (int)d_src_C1.height(),
            (int)d_src_C1.pitch());

    // The target image has 3/2 
    int trg_width = (src_rect.width*3+1) / 2;
    npp::ImageCPU_8u_C1 h_trg_C1(trg_width+1, src_rect.height);
    npp::ImageNPP_8u_C1 d_trg_C1(h_src_C1);

    Rectangle trg_rect(0, 0, (int)d_trg_C1.width(), (int)d_trg_C1.height(),
            (int)d_trg_C1.pitch());

    dim3 grid( (trg_rect.numElements() + X_STRIDE - 1) / X_STRIDE,
            trg_rect.height);
    dim3 tpb(X_STRIDE, 1);

    init_trg_kernel<<<grid, tpb>>>(d_src_C1.data(), d_trg_C1.data(),
            src_rect, trg_rect);
    
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    d_trg_C1.copyTo(h_trg_C1.data(), h_trg_C1.pitch());

    npp::saveImage(args.output_name, h_trg_C1);

    nppiFree(h_src_C1.data());
    nppiFree(d_src_C1.data());
    nppiFree(h_trg_C1.data());
    nppiFree(d_trg_C1.data());

    exit (0);
}

