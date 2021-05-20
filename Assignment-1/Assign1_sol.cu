
__global__ void per_row_kernel(int m,int n,int *A,int *B,int *C) 
{
    long long int total_no_of_threads=blockDim.x*blockDim.y*blockDim.z;
    long long int id=threadIdx.x + blockIdx.x * blockDim.x;
    for(long long int i=id;i<m;i+=total_no_of_threads)
    {
        for(long long int j=0;j<n;j++)
            C[i*n + j]=A[i*n + j] + B[i*n + j];
    }
}

__global__ void per_column_kernel(int m,int n,int *A,int *B,int *C) 
{
    long long int total_no_of_threads=blockDim.x*blockDim.y*blockDim.z;
    long long int id=blockDim.x*blockDim.y*blockIdx.x + blockDim.x*threadIdx.y + threadIdx.x;
    for(long long int i=id;i<n;i+=total_no_of_threads)
    {
        for(long long int j=0;j<m;j++)
            C[i+j*n]=A[i + j*n] + B[i + j*n];
    }
}

__global__ void per_element_kernel(int m,int n,int *A,int *B,int *C) 
{
    long long int total_no_of_threads=gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z;
    long long int blockid=gridDim.x * blockIdx.y + blockIdx.x;
    long long int id=blockDim.x*blockDim.y*blockid + blockDim.x*threadIdx.y + threadIdx.x;
    for(int i=id;i<m*n;i+=total_no_of_threads)
    {
        C[i]=A[i] + B[i];
    }
}

