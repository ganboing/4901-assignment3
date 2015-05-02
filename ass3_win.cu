/* 
   skeleton code for assignment3 COMP4901D
   Hash Join
   xjia@ust.hk 2015/04/15
 */
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <memory>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
using namespace std;

const int numBits = 6;
const int totalBits = 19;
const int numPart = 1 << numBits; // = 2^6
const int numPerPart = 1 << (totalBits - numBits); // = 2^(19-6)
const int mask = (1 << numBits) - 1;
const int numThreads = 128;
//const int numBlocks = 512;

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


/*
   return the partition ID of the input element
*/
  __device__
int getPartID(int element)
{
  element >>= (totalBits - numBits);
  return element & mask;
}

/*
	input: d_key[], array size N
	output: d_pixArray[]
	funciton: for input array d_key[] with size N, return the partition ID array d_pixArray[]
*/
  __global__
void mapPart(int d_key[],int d_pidArray[],int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadNumber = blockDim.x * gridDim.x;

  while(tid < N)
  {
	d_pidArray[tid] = getPartID(d_key[tid]);
	tid += threadNumber;
  }
}

/*
   input: d_pidArray[], array size N
   output: d_Hist[] 
   function: calculate the histogram d_Hist[] based on the partition ID array d_pidArray[]
*/
  __global__
void count_Hist(int d_Hist[],int d_pidArray[],int N)
{
  __shared__ int s_Hist[numThreads * numPart];
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int threadNumber = blockDim.x * gridDim.x;
  int offset = threadIdx.x * numPart;

  for(int i = 0; i < numPart; ++i)
	s_Hist[i + offset] = 0;

  for(int i = threadId; i < N; i += threadNumber)
	s_Hist[offset + d_pidArray[i]]++;

  for(int i = 0; i < numPart; ++i)
	d_Hist[i * threadNumber + threadId] = s_Hist[offset + i];
  __syncthreads();
}
/*
	input: d_pidArray[] (partition ID array), d_psSum[] (prefix sum of histogram), array size N
	output: d_loc[] (location array)
	function: for each element, calculate its corresponding location in the result array based on its partition ID and prefix sum of histogram
*/
  __global__
void write_Hist(int d_pidArray[],int d_psSum[],int d_loc[],int N)
{
  __shared__ int s_psSum[numThreads * numPart];
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int threadNumber = gridDim.x * blockDim.x;
  int offset = threadIdx.x * numPart;

  for(int i = 0; i < numPart; ++i)
	s_psSum[i + offset] = d_psSum[threadId + i * threadNumber];

  for(int i = threadId; i < N; i += threadNumber)
  {
	int pid = d_pidArray[i];
	d_loc[i] = s_psSum[pid + offset];
	s_psSum[pid + offset]++;
  }
}

/*
	input: d_psSum[] (prefix sum of histogram), array size N
	output: start position of each partition
	function: for each partition (chunck to be loaded in the join step), calculate its start position in the result array (the first element's position of this partition)
*/
  __global__
void getStartPos(int d_psSum[],int d_startPos[],int N)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int threadNumber = gridDim.x * blockDim.x;

  if(tid >= numPart)
	return;
  d_startPos[tid] = d_psSum[tid * threadNumber];
}

/*
    input: d_key[],d_value[],d_loc[],array size []
	output: out_key[],out_value[]
	function: rewrite the (key,value) pair to its corresponding position based on location array d_loc[]
*/
  __global__
void scatter(int d_key[],float d_value[],int out_key[],float out_value[],int d_loc[],int N)
{
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int threadNumber = blockDim.x * gridDim.x;

  while(threadId < N)
  {
	out_key[d_loc[threadId]] = d_key[threadId];
	out_value[d_loc[threadId]] = d_value[threadId];
	threadId += threadNumber;
  }
}

/*
	function: split the (key,value) array with size N, record the start position of each partition at the same time
*/
void split(int *d_key,float *d_value,int *d_startPos,int N)
{
  dim3 grid;
  dim3 block;
  if(N<numThreads){
      grid=1;
      block=N;
  }else{
      grid=(N+numThreads-1)/numThreads;
      block=numThreads;
  }
  int num_threads=grid.x * block.x;
  int hist_len = num_threads * numPart;
  int *d_pidArr, *d_Hist, *d_psSum, *d_loc, *d_outkey;
  float *d_outvalue;
  cudaMalloc(&d_outkey, sizeof(int)*N);
  cudaCheckError();
  cudaMalloc(&d_outvalue, sizeof(float)*N);
  cudaCheckError();
  cudaMalloc(&d_loc,sizeof(int)*N);
  cudaCheckError();
  cudaMalloc(&d_pidArr, sizeof(int)*N);
  cudaCheckError();
  cudaMalloc(&d_Hist, sizeof(int)*hist_len);
  cudaCheckError();
  cudaMalloc(&d_psSum, sizeof(int)*hist_len);
  cudaCheckError();

  mapPart<<<grid,block>>>(d_key, d_pidArr, N);
  cudaCheckError();
  count_Hist<<<grid,block>>>(d_Hist, d_pidArr, N);
  cudaCheckError();
  thrust::device_ptr<int> dev_Hist(d_Hist);
  thrust::device_ptr<int> dev_psSum(d_psSum);
  thrust::exclusive_scan(dev_Hist, dev_Hist + hist_len, dev_psSum);
  cudaCheckError();
  getStartPos<<<grid,block>>>(d_psSum, d_startPos, N);
  cudaCheckError();
  write_Hist<<<grid,block>>>(d_pidArr, d_psSum, d_loc, N);
  cudaCheckError();
  scatter<<<grid,block>>>(d_key, d_value, d_outkey, d_outvalue, d_loc, N);
  cudaCheckError();
  cudaMemcpy(d_key, d_outkey, sizeof(int)*N, cudaMemcpyDeviceToDevice);
  cudaCheckError();
  cudaMemcpy(d_value, d_outvalue, sizeof(float)*N, cudaMemcpyDeviceToDevice);
  cudaCheckError();

  cudaFree(d_psSum);
  cudaCheckError();
  cudaFree(d_Hist);
  cudaCheckError();
  cudaFree(d_pidArr);
  cudaCheckError();
  cudaFree(d_loc);
  cudaCheckError();
  cudaFree(d_outvalue);
  cudaCheckError();
  cudaFree(d_outkey);
  cudaCheckError();
  /* add your code here */
}

/*
	function: perform hash join on two (key,value) arrays 
*/   
  __global__
void join(int d_key1[],float d_value1[],int d_key2[],float d_value2[],int d_startPos1[],int d_startPos2[],int d_result[],int N1,int N2)
{
  /* add your code here */
}

void check_arr(int* arr, int N){
  int lower = std::numeric_limits<int>::min();
  std::for_each(arr, arr+N, [&](int& val){
    if(val < lower){
        fprintf(stderr, "array not sorted! @ %td\n", &val - arr);
        exit(-1);
    }
    else{
        lower = val;
    }
  });
}

void print_arr(int* arr, int* loc, int N){
  fprintf(stderr, "arr:\n");
  //check_arr(arr, N);
  for(int i=0;i<numPart;++i){
      int start=loc[i], end;
      if(i==numPart-1){
          end = N;
      }else{
          end = loc[i+1];
      }
      fprintf(stderr, "from %d to %d: ", start, end);
      for(int j=start;j!=end;++j){
          fprintf(stderr, "%08x ", arr[j]);
      }
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "loc:\n");
  for(int i=0;i<numPart;++i){
      fprintf(stderr, "%d ", loc[i]);
  }
  fprintf(stderr, "\n");
}

void hashJoin(int *d_key1,float *d_value1,int *d_key2,float *d_value2,int N1,int N2,int *d_result)
{
  int *d_startPos1,*d_startPos2;
  cudaMalloc(&d_startPos1,sizeof(int) * numPart);
  cudaCheckError();
  cudaMalloc(&d_startPos2,sizeof(int) * numPart);
  cudaCheckError();

  split(d_key1,d_value1,d_startPos1,N1);

  std::unique_ptr<int[]> arr1_finish(new int[N1]);
  std::unique_ptr<int[]> arr1_loc(new int[numPart]);
  cudaMemcpy(arr1_loc.get(), d_startPos1, sizeof(int)*numPart, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy(arr1_finish.get(), d_key1, sizeof(int)*N1, cudaMemcpyDeviceToHost);
  cudaCheckError();

  fprintf(stderr, "arr1: ");
  print_arr(arr1_finish.get(), arr1_loc.get(), N1);

  split(d_key2,d_value2,d_startPos2,N2);

  std::unique_ptr<int[]> arr2_finish(new int[N2]);
  std::unique_ptr<int[]> arr2_loc(new int[numPart]);
  cudaMemcpy(arr2_loc.get(), d_startPos2, sizeof(int)*numPart, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy(arr2_finish.get(), d_key2, sizeof(int)*N2, cudaMemcpyDeviceToHost);
  cudaCheckError();

  fprintf(stderr, "arr2: ");
  print_arr(arr2_finish.get(), arr2_loc.get(), N2);

  dim3 grid(numPart);
  dim3 block(1024);

  join<<<grid,block>>>(d_key1,d_value1,d_key2,d_value2,d_startPos1,d_startPos2,d_result,N1,N2);
}
int main()
{
  freopen("in.txt","r",stdin);
  int *h_key1, *h_key2, *d_key1, *d_key2;
  float *h_value1, *h_value2, *d_value1, *d_value2;
  int *h_result, *d_result;
  int N1,N2;

  {
    int tmp = scanf("%d%d",&N1,&N2);
    (void)tmp;
    assert(tmp==2);
  }

  h_key1 = (int*)malloc(N1 * sizeof(int));
  h_key2 = (int*)malloc(N2 * sizeof(int));
  h_value1 = (float*)malloc(N1 * sizeof(float));
  h_value2 = (float*)malloc(N2 * sizeof(float));
  h_result = (int*)malloc(N1 * sizeof(int));

  cudaMalloc(&d_key1, N1 * sizeof(int));
  cudaCheckError();
  cudaMalloc(&d_key2, N2 * sizeof(int));
  cudaCheckError();
  cudaMalloc(&d_value1, N1 * sizeof(float));
  cudaCheckError();
  cudaMalloc(&d_value2, N2 * sizeof(float));
  cudaCheckError();
  cudaMalloc(&d_result, N1 * sizeof(int));
  cudaCheckError();

  for(int i = 0; i < N1; ++i){
      int tmp = scanf("%d%f",&h_key1[i],&h_value1[i]);
      (void)tmp;
      assert(tmp==2);
  }

  for(int i = 0; i < N2; ++i){
      int tmp = scanf("%d%f",&h_key2[i],&h_value2[i]);
      (void)tmp;
      assert(tmp==2);
  }

  memset(h_result,-1,sizeof(int) * N1);
  cudaMemcpy(d_key1,h_key1, sizeof(int) * N1, cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(d_result,h_result, sizeof(int) * N1, cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(d_key2,h_key2, sizeof(int) * N2, cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(d_value1,h_value1, sizeof(float) * N1, cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(d_value2,h_value2, sizeof(float) * N2, cudaMemcpyHostToDevice);
  cudaCheckError();
  
  hashJoin(d_key1,d_value1,d_key2,d_value2,N1,N2,d_result);
  cudaCheckError();

  cudaMemcpy(h_result,d_result,sizeof(int) * N1, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy(h_key1,d_key1,sizeof(int) * N1, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy(h_key2,d_key2,sizeof(int) * N2, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy(h_value1,d_value1,sizeof(float) * N1, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaMemcpy(h_value2,d_value2,sizeof(float) * N2, cudaMemcpyDeviceToHost);
  cudaCheckError();

  int matched = 0;
  freopen("out.txt","w",stdout);
  for(int i = 0;i < N1; ++i)
  {
	if(h_result[i] == -1)
	  continue;
	matched++;
	printf("Key %d\nValue1 %.2f Value2 %.2f\n\n",h_key1[i],h_value1[i],h_value2[h_result[i]]);
  }
  printf("Matched %d\n",matched);
  fclose(stdout);
  freopen("/dev/tty","w",stdout);
  
  free(h_key1);
  free(h_key2);
  free(h_value1);
  free(h_value2);
  free(h_result);

  cudaFree(d_key1);
  cudaCheckError();
  cudaFree(d_key2);
  cudaCheckError();
  cudaFree(d_value1);
  cudaCheckError();
  cudaFree(d_value2);
  cudaCheckError();
  cudaFree(d_result);
  cudaCheckError();

  cudaDeviceReset();
  cudaCheckError();
  return 0;
}

