
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
    
    __host__ __device__ Rectangle():
        x0(0), y0(0), width(0), height(0), stride(0) {}
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

    if (argc!=4) {
        std::cerr << "Format:\n display_multires "
            "<NUM_RES>=1..9 <INPUT_FILE> <OUTPUT_FILE>\n";
        exit(-1);
    }
    //for (int ci=0; ci<argc; ci++) 
    //    std::cout << "arg " << ci << " " << std::string(argv[ci]) << std::endl;
    {
        args.k = atoi(argv[1]);
        assert(args.k>0 && args.k<10);
    }
    args.input_name = argv[2];
    args.output_name = argv[3];
    return args;
}


#define X_STRIDE 64

// copy src image to first 2/3rds of image, fill the rest with zeros.
__global__ void init_trg_kernel(const Npp8u* src, Npp8u* trg,
        Rectangle srcRect, Rectangle trgRect) {
    const int tidx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidx_y = blockDim.y * blockIdx.y + threadIdx.y;

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

// Take a source subrectangle of the image downscale it by a factor of X2 and output
// the result to a target subrectangle
__global__ void downscale_subimage_kernel(Npp8u* img,
        Rectangle fullRect, Rectangle srcRect, Rectangle trgRect) {
    const int tidx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidx_y = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ Npp8u buff[2][X_STRIDE];

    const int src_x = srcRect.x0 + tidx_x;
    const int src_y = srcRect.y0 + tidx_y;
    const int src_addr = src_y * fullRect.stride + src_x;

    if (tidx_x < srcRect.width && src_x < fullRect.width
            && tidx_y < srcRect.height && src_y < fullRect.height
            && src_addr < fullRect.numElements()) {
        buff[threadIdx.y][threadIdx.x] = img[src_addr];
    } else {
        buff[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x%2==0 && threadIdx.y==0) {
        buff[threadIdx.y][threadIdx.x] =
            (Npp8u) ( (
                ((int) buff[threadIdx.y+0][threadIdx.x+0])
                + ((int) buff[threadIdx.y+0][threadIdx.x+1])
                + ((int) buff[threadIdx.y+1][threadIdx.x+0])
                + ((int) buff[threadIdx.y+1][threadIdx.x+1])
                ) / 4 );
    }
    __syncthreads();

    const int trg_x = trgRect.x0 + tidx_x/2;
    const int trg_y = trgRect.y0 + tidx_y/2;
    const int trg_addr = trg_y * fullRect.stride + trg_x;

    if (threadIdx.x%2==0 && threadIdx.y==0
            && tidx_x/2 < trgRect.width && trg_x < fullRect.width
            && tidx_y/2 < trgRect.height && trg_y < fullRect.height
            && trg_addr < fullRect.numElements()) {
        img[trg_addr] = buff[threadIdx.y][threadIdx.x];
    }
}


// Calculates the next target rectangle, downscales the source subimage to it, and
// returns the target rectangle.
__host__ Rectangle downscale_subimage(Npp8u* img_ptr, Rectangle imgRect, Rectangle
        srcRect, int k) {
    Rectangle trgRect;
    // Rectangle(int x0, int y0, int width, int height, int pitch):
    if (!k) {
        // First rectangle is to the left of the original image;
        trgRect = Rectangle(
                srcRect.x_end(),    // new x0
                0,                  // new y0
                srcRect.width/2,    // new width
                srcRect.height/2,   // new height
                srcRect.stride);    // same stride.
    } else {
        // The next rectangles are placed downwards of each other
        trgRect = Rectangle(
                srcRect.x0 + srcRect.width/2,   // new x0
                srcRect.y_end(),                // new x0
                srcRect.width/2,                // new width
                srcRect.height/2,               // new height
                srcRect.stride);                // same stride.
    }

    std::cout <<
        "img ["
        << imgRect.x0 << ","
        << imgRect.y0 << ","
        << imgRect.width << ","
        << imgRect.height
        << "]: downscale ["
        << srcRect.x0 << ","
        << srcRect.y0 << ","
        << srcRect.width << ","
        << srcRect.height
        << "] -> ["
        << trgRect.x0 << ","
        << trgRect.y0 << ","
        << trgRect.width << ","
        << trgRect.height << "]" << std::endl;

    dim3 grid( (srcRect.numElements() + X_STRIDE - 1) / X_STRIDE,
            srcRect.height/2);
    dim3 tpb(X_STRIDE, 2);

    downscale_subimage_kernel<<<grid, tpb>>>(img_ptr,
            imgRect, srcRect, trgRect);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    return trgRect;
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
    npp::ImageCPU_8u_C1 h_trg_C1(trg_width, src_rect.height);
    npp::ImageNPP_8u_C1 d_trg_C1(h_trg_C1);

    Rectangle trg_rect(0, 0, (int)d_trg_C1.width(), (int)d_trg_C1.height(),
            (int)d_trg_C1.pitch());
    std::cout <<
        "img ["
        << src_rect.x0 << ","
        << src_rect.y0 << ","
        << src_rect.width << ","
        << src_rect.height
        << "] -> ["
        << trg_rect.x0 << ","
        << trg_rect.y0 << ","
        << trg_rect.width << ","
        << trg_rect.height << "]" << std::endl;

    dim3 grid( (trg_rect.numElements() + X_STRIDE - 1) / X_STRIDE,
            trg_rect.height);
    dim3 tpb(X_STRIDE, 1);

    init_trg_kernel<<<grid, tpb>>>(d_src_C1.data(), d_trg_C1.data(),
            src_rect, trg_rect);
    
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    Rectangle currRect = src_rect;
    for(int k=0; k<args.k; ++k) {
        if (currRect.width < 4 || currRect.height<4) break;
        currRect = downscale_subimage(
                d_trg_C1.data(), trg_rect, currRect, k);
    }

    d_trg_C1.copyTo(h_trg_C1.data(), h_trg_C1.pitch());

    npp::saveImage(args.output_name, h_trg_C1);

    nppiFree(h_src_C1.data());
    nppiFree(d_src_C1.data());
    nppiFree(h_trg_C1.data());
    nppiFree(d_trg_C1.data());

    exit (0);
}

