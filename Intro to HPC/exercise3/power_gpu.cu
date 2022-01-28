// the subroutine for GPU code can be found in several separated text file from the Brightspace.
// You can add these subroutines to this main code.
////////////////////////////////////////////


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"

int GlobalSize = 500;
const int BlockSize = 128;
const int BLOCK_SIZE =64;  // number of threads per block
// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;
float* d_Lamda = NULL;
float* h_Lamda = NULL;

// Variables to change
//int GlobalSize = 500;         // this is the dimension of the matrix, GlobalSize*GlobalSize
//const int BlockSize = 128;            // number of threads in each block
const float EPS = 0.000005;    // tolerence of the error
int max_iteration = 1000;       // the maximum iteration steps


// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void  Arguments(int, char**);
void checkCardVersion(void);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N);
__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N);
__global__ void ComputeLamda( float* g_VecV,float* g_VecW, float * g_Lamda,int N);

__global__ void Globalmem_Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void Globalmem_FindNormW(float* g_VecW, float* g_NormW, int N);
__global__ void Globalmem_NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV, int N);
__global__ void Globalmem_ComputeLamda( float* g_VecV,float* g_VecW, float* g_Lamda,int N);



void CPU_AvProduct()
{
	int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];

		}
	}
}

void CPU_NormalizeW()
{
	int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];

	normW = sqrt(normW);
	//printf("square root is: %f\n",normW);
	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
	int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];

	return lamda;
}

void RunCPUPowerMethod()
{
	fprintf(stderr,"*************************************\n");
	float oldLamda =0;
	float lamda=0;

	//AvProduct
	CPU_AvProduct();

	//power loop
	for (int i=0;i<max_iteration;i++)
	{
		CPU_NormalizeW();
		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
		//fprintf(stderr,"CPU lamda at %d: %f \n", i, lamda);
		//fprintf(stderr,"CPU tolerance at %d: %f \n", i, abs(oldLamda - lamda));
		// If residual is lass than epsilon break
		if(abs(oldLamda - lamda) < EPS)
			break;
		oldLamda = lamda;

	}
	fprintf(stderr,"*************************************\n");

}

// Host code
int main(int argc, char** argv)
{

    struct timespec t_start,t_end;
    double runtime;
		double memtime; //to meaasure the memory time
		struct timespec mt_start,mt_end; //variables for measuring memory time
    Arguments(argc, argv);

    int N = GlobalSize;
    fprintf(stderr,"Matrix size %d X %d \n", N, N);
		fprintf(stderr,"Threads per block %d \n",BlockSize);
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);

    // Allocate normalized value in host memory
    h_NormW = (float*)malloc(norm_size);
    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);

		h_Lamda = (float*)malloc(norm_size);

    // Initialize input matrix
    UploadArray(h_MatA, N);
    InitOne(h_VecV,N);


		fprintf(stderr,"Power method in CPU starts\n");
    clock_gettime(CLOCK_REALTIME,&t_start);
    RunCPUPowerMethod();   // the lamda is already solved here
    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    fprintf(stderr,"CPU: run time = %f secs.\n",runtime);
    fprintf(stderr,"Power method in CPU is finished\n");


    /////////////////////////////////////////////////
    // This is the starting points of GPU

    fprintf(stderr,"Power method in GPU starts\n");
    checkCardVersion();

    // Initialize input matrix
    InitOne(h_VecV,N);

    clock_gettime(CLOCK_REALTIME,&t_start);  // Here I start to count

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;
    int sharedMemSize = threadsPerBlock * threadsPerBlock * sizeof(float); // in per block, the memory is shared
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate matrix and vectors in device memory
		clock_gettime(CLOCK_REALTIME,&mt_start); //to measure the memory access time
    cudaMalloc((void**)&d_MatA, mat_size);
		//fprintf(stderr,"mallocs done\n");

    cudaMalloc((void**)&d_VecV, vec_size);
    cudaMalloc((void**)&d_VecW, vec_size); // This vector is only used by the device
    cudaMalloc((void**)&d_NormW, norm_size);
		cudaMalloc((void**)&d_Lamda, norm_size);

    //Copy from host memory to device memory
		float OldLamda = 2*EPS;
		*h_Lamda = 0;
		*h_NormW = 0;
    cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Lamda, h_Lamda, norm_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
		clock_gettime(CLOCK_REALTIME,&mt_end);
		memtime = (mt_end.tv_sec - mt_start.tv_sec) + 1e-9*(mt_end.tv_nsec - mt_start.tv_nsec);

		int k = 0;
		while(abs(OldLamda - *h_Lamda) >= EPS && k <= max_iteration){//abs(OldLamda - *h_Lamda) >= EPS &&
			Av_Product<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N); //Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
			cudaDeviceSynchronize(); //Needed, kind of barrier to sychronize all threads
			OldLamda = *h_Lamda;
			*h_Lamda = 0;
			clock_gettime(CLOCK_REALTIME,&mt_start);
			cudaMemcpy(d_Lamda, h_Lamda, norm_size, cudaMemcpyHostToDevice);
			clock_gettime(CLOCK_REALTIME,&mt_end);
			memtime += (mt_end.tv_sec - mt_start.tv_sec) + 1e-9*(mt_end.tv_nsec - mt_start.tv_nsec);
			ComputeLamda<<<blocksPerGrid, threadsPerBlock>>>(d_VecV, d_VecW, d_Lamda, N); //add shared memory too if needed
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_REALTIME,&mt_start);
			cudaMemcpy(h_Lamda, d_Lamda, norm_size, cudaMemcpyDeviceToHost);
			clock_gettime(CLOCK_REALTIME,&mt_end);
			memtime += (mt_end.tv_sec - mt_start.tv_sec) + 1e-9*(mt_end.tv_nsec - mt_start.tv_nsec);
			FindNormW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, N); //add shared memory too if needed
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_REALTIME,&mt_start);
			cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
			clock_gettime(CLOCK_REALTIME,&mt_end);
			memtime += (mt_end.tv_sec - mt_start.tv_sec) + 1e-9*(mt_end.tv_nsec - mt_start.tv_nsec);
			float root;
			root = sqrtf(*h_NormW);
			//printf("sqrt is %f\n",root);
			*h_NormW = root;
			clock_gettime(CLOCK_REALTIME,&mt_start);
			cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
			clock_gettime(CLOCK_REALTIME,&mt_end);
			memtime += (mt_end.tv_sec - mt_start.tv_sec) + 1e-9*(mt_end.tv_nsec - mt_start.tv_nsec);
			NormalizeW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, d_VecV, N); //add shared memory too if needed
			cudaDeviceSynchronize();
			*h_NormW = 0;
			clock_gettime(CLOCK_REALTIME,&mt_start);
			cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
			clock_gettime(CLOCK_REALTIME,&mt_end);
			memtime += (mt_end.tv_sec - mt_start.tv_sec) + 1e-9*(mt_end.tv_nsec - mt_start.tv_nsec);
			k++;
		}

		//use the below commented-out code for using the global memory implementation
		/*
		while(k <= max_iteration){//abs(OldLamda - *h_Lamda) >= EPS &&
			Globalmem_Av_Product<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
			cudaDeviceSynchronize(); //Needed, kind of barrier to sychronize all threads
			//OldLamda = *h_Lamda;
			*h_Lamda = 0;
			cudaMemcpy(d_Lamda, h_Lamda, norm_size, cudaMemcpyHostToDevice);
			Globalmem_ComputeLamda<<<blocksPerGrid, threadsPerBlock>>>(d_VecV, d_VecW, d_Lamda, N);
			cudaDeviceSynchronize();
			cudaMemcpy(h_Lamda, d_Lamda, norm_size, cudaMemcpyDeviceToHost);
			Globalmem_FindNormW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, N);
			cudaDeviceSynchronize();
			cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
			float root;
			root = sqrtf(*h_NormW);
			*h_NormW = root;
			//printf("sqrt is %f\n",root);
			cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
			Globalmem_NormalizeW<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, d_VecV, N);
			cudaDeviceSynchronize();
			//fprintf(stderr,"sqrt at %d: %f\n",k, root);
			fprintf(stderr,"GPU lamda at %d: %f \n", k, *h_Lamda);
			//fprintf(stderr,"tolerance at %d: %f \n", k, abs(OldLamda - *h_Lamda));
			*h_NormW = 0;
			cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
			k++;
		}*/




		fprintf(stderr,"*************************************\n");




    clock_gettime(CLOCK_REALTIME,&t_end);
    runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
    fprintf(stderr,"GPU: run time = %f secs.\n",runtime);
		fprintf(stderr,"GPU: memory access time = %f secs. \n",memtime);
		fprintf(stderr,"GPU: Pure computation time = %f secs. \n",runtime-memtime);
    // printf("Overall CPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_CPU));

    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
	  if (d_NormW)
		    cudaFree(d_NormW);
		if(d_Lamda)
			  cudaFree(d_Lamda);

    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
    if (h_NormW)
        free(h_NormW);

		if(h_Lamda)
			free(h_Lamda);

    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
   int total = n*n;
   int value=1;
    for (int i = 0; i < total; i++)
    {
    	data[i] = (int) (rand() % (int)(101));//1;//value;
	    value ++; if(value>n) value =1;
      // data[i] = 1;
    }
}

// Obtain program arguments
void Arguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
        {
            GlobalSize = atoi(argv[i+1]);
		    i = i + 1;
        }
        if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0)
        {
            max_iteration = atoi(argv[i+1]);
		    i = i + 1;
        }
    }
}


void checkCardVersion()
{
   cudaDeviceProp prop;

   cudaGetDeviceProperties(&prop, 0);

   fprintf(stderr,"This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
   if(prop.major < 2)
   {
      fprintf(stderr,"Need compute capability 2 or higher.\n");
      exit(1);
   }
}

//GPU Functions
/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

// ****************************************************************************************************************************************************/
// parallelization method for the Matrix-vector multiplication as follows:

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of matrix A or vector V from global memory to share memory,
// a square submatrix of A is shared by a block, the size of square submatrix is BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to
// handle a multiplication of each row of Matrix A and vector V step by step. In eacg step, two subvectors with size BLOCK_SIZE is multiplied.
//*****************************************************************************************************************************************************/


__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int aBegin = N * BlockSize * bx;

    int aEnd   = aBegin + N - 1;
    int step  = BlockSize;

    int bBegin = 0;//BLOCK_SIZE * bx;
    int bIndex=0;
    int aIndex =0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += step, b += step)
    {

        __shared__ float As[BlockSize*BlockSize];


        __shared__ float bs[BlockSize];



        for (int aa = 0; aa < BlockSize;aa+= 1)
        {
            aIndex = a+tx+aa*N;
            if( aIndex < N*N)
        	    As[tx+aa*BlockSize] = g_MatA[aIndex];
		        else
        	    As[tx+aa*BlockSize] = 0;
        }

        bIndex = b+tx;
   	    if(bIndex<N)
		      bs[tx] = g_VecV[bIndex];
	      else
		      bs[tx] = 0;

        __syncthreads();

        for (int k = 0; k < BlockSize; ++k)
        {
            Csub += As[k+tx*BlockSize] * bs[k];
        }//}
        __syncthreads();
    }

    g_VecW[ BlockSize * bx + tx] = Csub;
}


/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[];
	//extern __device__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] =  g_VecW[globalid];
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  sdata[tid] = sdata[tid] * sdata[tid];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_NormW,sdata[0]);
}


__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sNormData[];
	//extern __device__ float sNormData[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  if(tid==0) sNormData[0] =  g_NormW[0];
  __syncthreads();

  // For thread ids greater than data space
  if (globalid < N) {
     g_VecV[globalid] = g_VecW[globalid]/sNormData[0];
  }

}

__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdataVW[];
	//extern __device__ float sdataVW[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdataVW[tid] =  g_VecV[globalid] * g_VecW[globalid];
  }
  else {
     sdataVW[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdataVW[tid] = sdataVW[tid] + sdataVW[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_Lamda,sdataVW[0]);
}


//equivalent functions making use of global memory instead of shared memory
__global__ void Globalmem_Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int globalIndex = bx*BlockSize+tx; //this is the row number that current thread works on
	if(globalIndex < N){ //if its not an extra thread that lies beyond the matrix dimensions
		int a_start_index = globalIndex*N;
		float sum = 0.0;
		for(int i = 0;i < N;i++){
			sum = sum + g_MatA[a_start_index + i] * g_VecV[i];
		}
		g_VecW[globalIndex] = sum;
	}
}

__global__ void Globalmem_FindNormW(float* g_VecW, float* g_NormW, int N)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int globalIndex = bx*BlockSize+tx; //this is the row number that current thread works on
	if(globalIndex < N){
		float squareTerm = g_VecW[globalIndex]*g_VecW[globalIndex];
		atomicAdd(g_NormW,squareTerm);
	}
}




__global__ void Globalmem_ComputeLamda(float* g_VecV, float* g_VecW, float* g_Lamda,int N)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int globalIndex = bx*BlockSize+tx; //this is the row number that current thread works on
	if(globalIndex<N){
		float product = g_VecV[globalIndex]*g_VecW[globalIndex];
		atomicAdd(g_Lamda,product);
	}
}

__global__ void Globalmem_NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV,int N)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int globalIndex = bx*BlockSize+tx; //this is the row number that current thread works on
	if(globalIndex<N){
			g_VecV[globalIndex] = g_VecW[globalIndex]/g_NormW[0];
	}
}
