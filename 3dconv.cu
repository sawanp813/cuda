#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Program-wide constants
#define O_TILE_WIDTH 8 // Width of 'output' blocks for bound-checking
#define MASK_WIDTH 3
#define BLOCK_WIDTH (O_TILE_WIDTH + (MASK_WIDTH - 1)) // Width of thread blocks

// Constant memory for device kernel
__constant__ float M[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH]; // 3D Mask

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  printf("%s \n", "Entered kernel");
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int depth_o = blockIdx.z * O_TILE_WIDTH + tz; 
  int row_o = blockIdx.y * O_TILE_WIDTH + ty;
  int col_o = blockIdx.x * O_TILE_WIDTH + tx;

  int depth_i = depth_o - MASK_WIDTH/2;
  int row_i = row_o - MASK_WIDTH/2;
  int col_i = col_o - MASK_WIDTH/2;

  __shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  if ((depth_i >= 0) && (depth_i < z_size) && (row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size)) {
    N_ds[tz][ty][tx] = input[depth_i * y_size * x_size + row_i * x_size + col_i];
  } else {
    N_ds[tz][ty][tx] = 0.0;
  }

  __syncthreads();

  float result = 0.0;

  if ((tz < O_TILE_WIDTH) && (ty < O_TILE_WIDTH) && (tx < O_TILE_WIDTH)) {
    for (int i = 0; i < MASK_WIDTH; i++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        for (int k = 0; k < MASK_WIDTH; k++) {
          result += (M[i * MASK_WIDTH * MASK_WIDTH + j * MASK_WIDTH + k] * N_ds[i+tz][j+ty][k+tx]);
        }
      }
    }

    if (depth_o < z_size && row_o < y_size && col_o < x_size) {
      output[depth_o * y_size * x_size + row_o * x_size + col_o] = result;
    }
  }
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  // inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  int sizeOutput = sizeof(float)*z_size*y_size*x_size;
  int sizeInput = sizeof(float)*z_size*y_size*x_size;
  cudaMalloc((void **) &deviceInput, sizeInput); // Allocating space for input matrix on device
  cudaMalloc((void **) &deviceOutput, sizeOutput); // Allocating space for output matrix on device
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  // first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, sizeInput, cudaMemcpyHostToDevice); // Copying input to device
  cudaMemcpyToSymbol(M, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float)); // Copying kernel to device (constant memory)
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  int gridCols = ceil(x_size/double(O_TILE_WIDTH));
  int gridRows = ceil(y_size/double(O_TILE_WIDTH));
  int gridDepth = ceil(z_size/double(O_TILE_WIDTH));
  dim3 DimGrid(gridCols, gridRows, gridDepth); 
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

  printf("%d \n", gridCols);
  printf("%d \n", gridRows);
  printf("%d \n", gridDepth);

  printf("%s \n", "About to launch kernel");
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  printf("%s \n", "kernel returned");
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput+3, deviceOutput, sizeOutput, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
