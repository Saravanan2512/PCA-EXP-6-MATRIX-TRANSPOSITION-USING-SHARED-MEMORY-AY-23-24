# PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME : karthikeyan k</h3>
<h3>ENTER YOUR REGISTER NO : 212223230101</h3>
<h3>EX. NO : 6</h3>
<h3>DATE : 17/11/2025</h3>
<h1> <align=center> MATRIX TRANSPOSITION USING SHARED MEMORY </h3>
  Implement Matrix transposition using GPU Shared memory.</h3>

## AIM:
To perform Matrix Multiplication using Transposition using shared memory.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
 CUDA_SharedMemory_AccessPatterns:

1. Begin Device Setup
    1.1 Select the device to be used for computation
    1.2 Retrieve the properties of the selected device
2. End Device Setup

3. Begin Array Size Setup
    3.1 Set the size of the array to be used in the computation
    3.2 The array size is determined by the block dimensions (BDIMX and BDIMY)
4. End Array Size Setup

5. Begin Execution Configuration
    5.1 Set up the execution configuration with a grid and block dimensions
    5.2 In this case, a single block grid is used
6. End Execution Configuration

7. Begin Memory Allocation
    7.1 Allocate device memory for the output array d_C
    7.2 Allocate a corresponding array gpuRef in the host memory
8. End Memory Allocation

9. Begin Kernel Execution
    9.1 Launch several kernel functions with different shared memory access patterns (Use any two patterns)
        9.1.1 setRowReadRow: Each thread writes to and reads from its row in shared memory
        9.1.2 setColReadCol: Each thread writes to and reads from its column in shared memory
        9.1.3 setColReadCol2: Similar to setColReadCol, but with transposed coordinates
        9.1.4 setRowReadCol: Each thread writes to its row and reads from its column in shared memory
        9.1.5 setRowReadColDyn: Similar to setRowReadCol, but with dynamic shared memory allocation
        9.1.6 setRowReadColPad: Similar to setRowReadCol, but with padding to avoid bank conflicts
        9.1.7 setRowReadColDynPad: Similar to setRowReadColPad, but with dynamic shared memory allocation
10. End Kernel Execution

11. Begin Memory Copy
    11.1 After each kernel execution, copy the output array from device memory to host memory
12. End Memory Copy

13. Begin Memory Free
    13.1 Free the device memory and host memory
14. End Memory Free

15. Reset the device

16. End of Algorithm

## PROGRAM:
```
%%writefile mattranpose.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif // _COMMON_H

#define BDIMX 16
#define BDIMY 16
#define IPAD  2

void printData(const char *msg, int *in, const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%4d", in[i]);
    }

    printf("\n\n");
}

/*
 * Kernels
 *
 * Note: use correct CUDA keywords __global__ and __shared__
 */

// store row-major, read row-major
__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // store
    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    // load
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

// store transposed in shared mem and read transposed (col->col)
__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // store
    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    // load
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

// alternate index math version (store into transposed indices)
__global__ void setColReadCol2(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[icol][irow] = idx;

    __syncthreads();

    out[idx] = tile[icol][irow];
}

// store row, read column (transposed read)
__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[icol][irow];
}

// padded shared memory to avoid bank conflicts
__global__ void setRowReadColPad(int *out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[icol][irow];
}

// dynamic shared memory variant (linearized)
__global__ void setRowReadColDyn(int *out)
{
    extern __shared__ int tile[]; // size = BDIMX*BDIMY

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    unsigned int col_idx = icol * blockDim.x + irow;

    tile[idx] = idx;

    __syncthreads();

    out[idx] = tile[col_idx];
}

// dynamic shared memory variant with padding to avoid bank conflicts
__global__ void setRowReadColDynPad(int *out)
{
    extern __shared__ int tile[]; // size = (BDIMX + IPAD) * BDIMY

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    tile[row_idx] = g_idx;

    __syncthreads();

    out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
    // device
    int dev = 0;
    double iStart, iElaps;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
    const char *bankMode = (pConfig == cudaSharedMemBankSizeFourByte) ? "4-Byte" :
                           (pConfig == cudaSharedMemBankSizeEightByte) ? "8-Byte" :
                           "Default";
    printf("Shared Mem Bank Mode: %s\n", bankMode);

    // array size (single block 16x16)
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = true;
    if (argc > 1) iprintf = atoi(argv[1]) != 0;

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d) >>>\n", grid.x, grid.y, block.x, block.y);

    // allocate device memory
    int *d_C = NULL;
    iStart = seconds();
    CHECK(cudaMalloc((void**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CHECK(cudaMemset(d_C, 0, nBytes));

    // setRowReadRow
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setRowReadRow       ", gpuRef, nx * ny);

    // setColReadCol
    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setColReadCol       ", gpuRef, nx * ny);

    // setColReadCol2
    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol2<<<grid, block>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setColReadCol2      ", gpuRef, nx * ny);

    // setRowReadCol
    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setRowReadCol       ", gpuRef, nx * ny);

    // setRowReadColDyn (dynamic shared mem size = BDIMX*BDIMY*sizeof(int))
    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setRowReadColDyn    ", gpuRef, nx * ny);

    // setRowReadColPad
    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setRowReadColPad    ", gpuRef, nx * ny);

    // setRowReadColDynPad (dynamic shared mem size = (BDIMX+IPAD)*BDIMY*sizeof(int))
    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("setRowReadColDynPad ", gpuRef, nx * ny);

    iElaps = seconds() - iStart;
    printf("Elapsed time %f sec\n", iElaps);

    // cleanup
    CHECK(cudaFree(d_C));
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
```

## OUTPUT:
<img width="1733" height="407" alt="image" src="https://github.com/user-attachments/assets/8b37b2cd-3059-4dfc-89e7-830b1e653eca" />


## RESULT:
Thus the program has been executed by using CUDA to transpose a matrix. It is observed that there are variations shared memory and global memory implementation. The elapsed times are recorded as 0.001042 sec.
