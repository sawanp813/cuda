// Histogram Equalization

#include <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
		        }                                                                     \
	    } while(0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256 

__global__ void convert_float_to_uchar(float* input, unsigned char* output, int size) {

  // collect tid which also is index for one pixel in image
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // if statement in case tid is out of bounds 
  if (tid < size) {

    output[tid] = (unsigned char) (255 * input[tid]);

  }
}

__global__ void rgb_to_grayscale(unsigned char* input, unsigned char* output, int size) {

  // define pointers
  unsigned char r;
  unsigned char g;
  unsigned char b;

  // collect tid, each thread handles conversion for one pixel
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // if statement in case tid is out of bounds
  if (tid < size) {

    r = (float) input[3*tid];
    g = (float) input[3*tid + 1];
    b = (float) input[3*tid + 2];

    // conversion statement
    output[tid] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void hist(unsigned char* buffer, unsigned int* histo, int size) {

  // Create block-level shared histogram array
  __shared__ unsigned int p_hist[HISTOGRAM_LENGTH];

  // initialize all values to 0
  p_hist[threadIdx.x] = 0;

  __syncthreads();

  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int numThreads = blockDim.x * gridDim.x;

  // add grayscale image pixel value to corresponding bin in shared histogram
  while (ind < size) {

    atomicAdd(&(p_hist[buffer[ind]]), 1);
    ind += numThreads;  // in case there aren't enough threads in grid for each pixel

  }

  __syncthreads();

  // add block-level histogram values to global histogram
  atomicAdd(&(histo[threadIdx.x]), p_hist[threadIdx.x]);

}

__global__ void calcCDF(float* cdf, unsigned int* histo, int imageWidth, int imageHeight, int length) {
  // calculates cdf of grayscale image pixel histogram

  // shared memory array
  __shared__ float partialScan[2*HISTOGRAM_LENGTH];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // populating of shared memory array with normalized initial values from global histogram
  if (tid < length) {

    partialScan[tid] = (float) histo[tid] / (float)(imageWidth * imageHeight);

  }

  __syncthreads();

  // Reduction step
  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
    if (index < length) {

      partialScan[index] += partialScan[index - stride];

    }
    __syncthreads();
  }

  // Post scan step
  for (unsigned int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
    __syncthreads();
    unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
    if (index + stride < length) {

      partialScan[index + stride] += partialScan[index];

    }
  }

  __syncthreads();

  if (tid < length) {

    cdf[tid] += partialScan[threadIdx.x];
    
  }
}

__global__ void histEqualization(unsigned char* ucharImage, float* cdf, int size) {
  // applies histogram equalization function to histogram cdf

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float cdfmin = cdf[0]; // minimum value of cdf from previous step

  // rescaling cdf values of grayscale image pixel histogram
  if (tid < size) {

    float x = 255.0F * (cdf[ucharImage[tid]] - cdfmin) / (1.0F - cdfmin);
    float start = 0.0F;
    float end = 255.0F;

    // performs 'clamping'
    if (start > x) {

      x = start;

    }
    if (end < x) {

      x = end;

    }

    // output final scaled value
    ucharImage[tid] = (unsigned char) x;
  }
}

__global__ void convert_uchar_to_float(unsigned char* input, float* output, int size) {
  // converts uchar image back to float

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < size) {

    output[tid] = (float) (input[tid] / 255.0);

  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  float* deviceInputImage;
  float* deviceOutputImage;
  unsigned int* histo;
  unsigned char* uCharImage;
  unsigned char* grayImage;
  float* cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  int size = imageWidth*imageHeight*imageChannels;

  // Allocating GPU memory
	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInputImage, size*sizeof(float)));
	wbCheck(cudaMalloc((void **)&uCharImage, size*sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **)&grayImage, imageWidth*imageHeight*sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **)&histo, HISTOGRAM_LENGTH*sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH*sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutputImage, size*sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	// Copying memory to the GPU here
	wbTime_start(Copy, "Copying input image to GPU.");
  cudaMemcpy(deviceInputImage, hostInputImageData,size*sizeof(float),cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying input image to GPU.");
	wbTime_start(Copy, "Zeroing out histogram.");
	wbCheck(cudaMemset(histo, 0, HISTOGRAM_LENGTH*sizeof(unsigned int)));
	wbTime_stop(Copy, "Zeroing out histogram.");
	wbTime_start(Copy, "Zeroing out cdf.");
	wbCheck(cudaMemset(cdf, 0, HISTOGRAM_LENGTH*sizeof(float)));
	wbTime_stop(Copy, "Zeroing out cdf.");

	// Initializing the grid and block dimensions
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((size - 1) / BLOCK_SIZE + 1);
	dim3 dimGridHist((size - 1) / HISTOGRAM_LENGTH + 1);

	// launch kernel to convert image from float to uchar values
	wbTime_start(Compute, "Performing CUDA mask to uCharImage.");
	convert_float_to_uchar << <dimGrid, dimBlock >> >(deviceInputImage, uCharImage, imageWidth*imageHeight*imageChannels);
	wbTime_stop(Compute, "Performing CUDA mask to uCharImage.");

	// launch kernel to convert RGB image to grasycale
	wbTime_start(Compute, "Performing CUDA converstion to grayImage.");
	rgb_to_grayscale << <dimGrid, dimBlock >> >(uCharImage, grayImage, imageWidth*imageHeight);
	wbTime_stop(Compute, "Performing CUDA converstion to grayImage.");

	// launch kernel to calculate histogram CDF
	wbTime_start(Compute, "Performing CUDA CDF histogram.");
	hist << <dimGrid, dimBlock >> >(grayImage, histo, imageWidth*imageHeight);
	wbTime_stop(Compute, "Performing CUDA CDF histogram.");

	// launch kernel to compute the CDF function of histogram
	wbTime_start(Compute, "Calculating CDF.");
	calcCDF << <dimGridHist, dimBlock >> >(cdf, histo, imageWidth, imageHeight, HISTOGRAM_LENGTH);
	wbTime_stop(Compute, "Calculating CDF.");

	// launch kernel to apply the histogram equalization function
	wbTime_start(Compute, "Performing CUDA histogram equalization function.");
	histEqualization << <dimGrid, dimBlock >> >(uCharImage, cdf, imageWidth*imageHeight*imageChannels);
	wbTime_stop(Compute, "Performing CUDA histogram equalization function.");

	// launch kernel to convert image from float to uchar 
	wbTime_start(Compute, "Performing CUDA from float to unsigned char.");
	convert_uchar_to_float << <dimGrid, dimBlock >> >(uCharImage, deviceOutputImage, imageWidth*imageHeight*imageChannels);
	wbTime_stop(Compute, "Performing CUDA from float to unsigned char.");

	// copy the GPU memory back to the CPU 
	wbCheck(cudaMemcpy(hostOutputImageData,
		deviceOutputImage,
		imageWidth*imageHeight*imageChannels*sizeof(float),
		cudaMemcpyDeviceToHost));

	// Output image format 
	wbImage_setData(outputImage, hostOutputImageData); 

	// free GPU memory here
	wbCheck(cudaFree(deviceInputImage));
	wbCheck(cudaFree(uCharImage));
	wbCheck(cudaFree(grayImage));
	wbCheck(cudaFree(histo));
	wbCheck(cudaFree(cdf));
	wbCheck(cudaFree(deviceOutputImage));

  wbSolution(args, outputImage);

  return 0;
}
