#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<iostream>
#include <sys/time.h> 
#include<bits/stdc++.h>
using namespace std;

struct edgepairs{
  int x;
  int y;
};

bool compareTwoEdgePairs(edgepairs a, edgepairs b)
{
    if (a.x != b.x)
        return a.x < b.x;

    if (a.y != b.y)
        return a.y < b.y;
 
  return true;
}

// complete the following kernel...
__global__ void dkernel_Adds(int *gpuOA, int *gpuCA, int *gpulocals,int *gpucurrentupdate,int n){
    unsigned int idx=blockIdx.x;
    __shared__ int  val;
    val=gpulocals[idx];
    int start,end;
    for(int i=0;i<n;i++)
    {
        start=gpuOA[i];
        end=gpuOA[i+1];
        for(int j=start;j<end;j++)
        {
            if(gpuCA[j]==idx)
            {
                val+=gpucurrentupdate[i];
            }
        }
    }
    gpulocals[idx]=val;
    
}

// complete the following kernel...
__global__ void dkernel_Mins(int *gpuOA, int *gpuCA, int *gpulocals,int *gpucurrentupdate,int n){
    unsigned int idx=blockIdx.x;
    __shared__ int  val;
    val=gpulocals[idx];
    int start,end;
    for(int i=0;i<n;i++)
    {
        start=gpuOA[i];
        end=gpuOA[i+1];
        for(int j=start;j<end;j++)
        {
            if(gpuCA[j]==idx)
            {
                val=min(gpucurrentupdate[i],val);
            }
        }
    }
    gpulocals[idx]=val;
}

// complete the following kernel...
__global__ void dkernel_Maxs(int *gpuOA, int *gpuCA, int *gpulocals,int *gpucurrentupdate,int n){
  unsigned int idx=blockIdx.x;
    __shared__ int  val;
    val=gpulocals[idx];
    int start,end;
    for(int i=0;i<n;i++)
    {
        start=gpuOA[i];
        end=gpuOA[i+1];
        for(int j=start;j<end;j++)
        {
            if(gpuCA[j]==idx)
            {
                val=max(gpucurrentupdate[i],val);
            }
        }
    }
    gpulocals[idx]=val;
}


int main(int argc,char **argv){

	//variable declarations
	int m,n;
	int number;
	int numofquery;
	int op;
	struct timeval t1, t2;
	vector <double> kerneltime;

	//File pointer declaration
	FILE *filePointer;

	//File Opening for read
	char *filename = argv[1]; 
    	filePointer = fopen( filename , "r") ; 
      
	//checking if file ptr is NULL
    	if ( filePointer == NULL ) 
    	{
        printf( "input.txt file failed to open." ) ; 
	      return 0;
    	}

	fscanf(filePointer, "%d", &n );		//scaning the number of vertices
        fscanf(filePointer, "%d", &m );		//scaning the number of edges

	//D.S to store the input graph in COO format
	vector <edgepairs> COO(m);
	
	//Reading from file and populate the COO
	for(int i=0 ; i<m ; i++ )
        {
		for(int j=0;j<2;j++){
			if ( fscanf(filePointer, "%d", &number) != 1)
            			break;
		if( j%2 == 0) 
		{       		
			if(number >= 1 && number <= 10000)
			COO[i].x = number;
		}		
		else
		{
			if(number >= 1 && number <= 10000)
			COO[i].y = number;
		}	

		}
        }
	// COO done...
	
	// sort the COO 
	sort(COO.begin(),COO.end(),compareTwoEdgePairs);
	//sorting COO done..
	
	// Converting the graph in COO format to CSR format..
	
	// create the CSR
	
	int *OA = (int *)malloc( (n+1)*sizeof(int));		//Offsets Array
	for(int i=0;i<n+1;i++){
                OA[i] = 0;
        }

	int *CA = (int *)malloc(m*sizeof(int));			//Coordinates Array
	OA[0]=0;

	//initialize the Coordinates Array
	for(int i=0;i<m;i++){
		if(COO[i].y >= 1 && COO[i].y <= 10000)
		CA[i] = COO[i].y - 1;
	}
	//initialize the Offsets Array
	for(int i=0;i<m;i++){
		if(COO[i].x >= 1 && COO[i].x <= 10000)
		OA[COO[i].x]++;		//store the frequency..
	}
	for(int i=0;i<n;i++){
		OA[i+1] += OA[i];	// do cumulative sum..
	}

	// Converting the graph to CSR done..
	
	// copy initial local values to the array from the file
	int *initlocalvals = (int *)malloc(n*sizeof(int));;
	for(int i=0 ; i<n ; i++ )
        {
        if ( fscanf(filePointer, "%d", &number) != 1)
            break;
         
        initlocalvals[i] = number;
        }
	// copying local vals end..

	// get number of queries from the file
	fscanf(filePointer, "%d", &numofquery);
	
	//copy OA,CA and initlocalvals to the GPU Memory
	int *gpuOA, *gpuCA, *gpulocals;
  cudaMalloc( &gpuOA, sizeof(int) * (1+n) );
  cudaMalloc( &gpuCA, sizeof(int) * m );
  cudaMalloc( &gpulocals, sizeof(int) * n );
	cudaMemcpy(gpuOA, OA, sizeof(int) * (1+n), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuCA, CA, sizeof(int) * m, cudaMemcpyHostToDevice);
	cudaMemcpy(gpulocals, initlocalvals, sizeof(int) * n, cudaMemcpyHostToDevice);


	int *currentupdate = (int *)malloc(n*sizeof(int));	// array to store the updates that are pushed by each vertex to there neighbors
	int *gpucurrentupdate;		// same as above but on GPU
  cudaMalloc( &gpucurrentupdate, sizeof(int) * n );
  int *results = (int *)malloc(n*sizeof(int));         // storing the results from GPU to CPU for the enumerate query


  // open the output.txt to write the query results
      char *fname = argv[2]; 
      FILE *fptr;
      fptr = fopen(fname,"w");

	for(int i=0;i<numofquery;i++){

		//read the operator
		fscanf(filePointer, "%d", &op);

		if(op != 3){					// if operator is other then enumerate (i.e. +,min,max)

		// read the current updates in the array				
			for(int j=0 ; j<n ; j++ )
	        	{
	        	 	if ( fscanf(filePointer, "%d", &number) != 1)
	            	 	break;
	        		currentupdate[j] = number;
	        	}

		// copy current updates to gpu
		cudaMemcpy(gpucurrentupdate, currentupdate, sizeof(int) * n, cudaMemcpyHostToDevice);
		//kernel launches
    if(op == 0)	{
		gettimeofday(&t1, 0);	
		dkernel_Adds<<<n,1>>>(gpuOA,gpuCA,gpulocals,gpucurrentupdate,n);
		cudaDeviceSynchronize();
		gettimeofday(&t2, 0);
		}
    if(op == 1)	{
		gettimeofday(&t1, 0);
		dkernel_Mins<<<n,1>>>(gpuOA,gpuCA,gpulocals,gpucurrentupdate,n);
		cudaDeviceSynchronize();
		gettimeofday(&t2, 0);
		}
    if(op == 2)	{	
		gettimeofday(&t1, 0);
		dkernel_Maxs<<<n,1>>>(gpuOA,gpuCA,gpulocals,gpucurrentupdate,n);
		cudaDeviceSynchronize();
		gettimeofday(&t2, 0);
		}
    
		double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0; // Time taken by kernel in seconds 
		kerneltime.push_back(time);  

    		printf("Time taken by kernel to execute is: %.6f ms\n", time); 
		}

		else{						// if operator is enumnerate then store the results to file
			//print local values of each vertices.
      cudaMemcpy(results, gpulocals, n * sizeof(int), cudaMemcpyDeviceToHost);  // get each locals from GPU
       for(int j=0;j<n;j++){
           fprintf(fptr ,"%d ", results[j] ); 
       }
       fprintf(fptr,"\n"); 
    }
		
	}

	int nall = kerneltime.size();
	double sumtime=0;
	for(int i=0;i<nall;i++){
		sumtime += kerneltime[i];
	}
	// print the time taken by all the kernels of the current test-case
	cout << "\ntotal time taken by the current test-case is " << sumtime << " ms\n";

  fclose(fptr);
  fclose(filePointer);

	return 0;
}
