//%%cu
#include <iostream>
#include <cuda.h>
#include<bits/stdc++.h>
using namespace std;

float device_time_taken;

struct edgepairs{
  int x;
  int y;
  int wt;
};

bool compareTwoEdgePairs(edgepairs a, edgepairs b)
{
    if (a.x != b.x)
        return a.x < b.x;

    if (a.y != b.y)
        return a.y < b.y;
 
  return true;
}

struct Graph{
    int nodes ;
    int edges;
    int *OA ;
    int *CA ;
    int *weight ;
    Graph(int n,int e){
        nodes = n ;
        edges = e ;
        OA = new int[nodes +1];
        CA = new int[2 * edges +1];
        weight = new int[2 * edges];
    }
};



__device__ void print(float* bcW,int node_count)
{
    for(int i=0;i<node_count;i++)
    {
        printf("%f ",bcW[i]) ;
    }
    printf("\n");
}
__global__ void cal_delta(Graph *graph , int *delta)
{
    int idx = threadIdx.x ;
    int min_wt = INT_MAX ;
    for(int i = graph->OA[idx] ; i < graph->OA[idx+1] ; i++)
    {
        min_wt = min(min_wt,graph->weight[i]);
    }
    delta[idx] = min_wt ;
}



__global__ void kernel(Graph *graph , float *BC , int *delta_node , int node_count) 
{
    int idx = threadIdx.x;
    extern __shared__ int array[];
    int *U = (int*)array ;
    int *F = (int*)&U[node_count] ;
    int *d = (int*)&F[node_count] ;
    int *sigma = (int*)&d[node_count] ;
    float *dependency = (float*)&sigma[ node_count ];
    int *lock = (int*)&dependency[node_count] ;
    int *ends = (int*)&lock[node_count] ;
    int *S = (int*)&ends[node_count] ;

    
    int v = threadIdx.x;  
    
    __shared__ int s;
    __shared__ int ends_len ;
    __shared__ int s_len ;
    __shared__ int delta ;

    s =  blockIdx.x ;   // Source vertex
    __syncthreads();


    // Initialisation
    if(v == s)
    {
        d[v] = 0 ;
        sigma[v] = 1 ;
        U[v] = 0 ;
        F[v] = 1 ;
        S[0] = s ;
        s_len = 1 ;
        ends[0] = 0 ;
        ends[1]  = 1 ;
        ends_len = 2 ;
    }
    else
    {
        U[v] = 1 ;
        F[v] = 0 ;
        d[v] = INT_MAX ;
        sigma[v] = 0 ;
    }
    dependency[v] = 0.0;
    lock[v] = 0 ;
    delta = 0 ;
    
 
    __syncthreads();
        
    // shortest path algorithm   
    while(delta < INT_MAX)
    {
        
        __syncthreads();
        if(F[v] == 1)
        {
            
            for(int r = graph->OA[v]; r < graph->OA[v + 1]; r++)
            {
                int w = graph->CA[r];
                int wt_vw = graph->weight[r] ;
                bool needlock = true ;
                while(needlock)
                {
                    if(atomicCAS(&lock[w],0,1) == 0)
                    {
                        if( U[w]== 1 && d[v] + wt_vw < d[w])
                        {
                            d[w] = d[v] + wt_vw ;
                            sigma[w] = 0 ;
                        }
                        if ( d[w] == d[v] + wt_vw )
                        {
                            sigma[w] = sigma[w] + sigma[v] ;
                        }
                        atomicExch(&lock[w] , 0 );
                        needlock = false ;
                    }
                }     
            }    
        }
      
        if(idx == 0)
        {
            atomicExch(&delta,INT_MAX);
        }
        __syncthreads() ;
        if( U[v] == 1 && d[v] < INT_MAX )
        {
            atomicMin( &delta , d[v] + delta_node[v] ) ;
        }
        __shared__ int count ;
        if(idx == 0)
        {
            atomicExch(&count,0);
        }
        F[v] = 0 ;
        __syncthreads() ;

            
        if(U[v] == 1 && d[v] < delta)
        {
            U[v] = 0;
            F[v] = 1;
            int t = atomicAdd(&s_len,1);
            S[t] = v ;
            atomicAdd(&count,1);
        }
        
        __syncthreads();
        
        if(idx == 0 )
        {
            if(count > 0) 
            {

                ends[ends_len] = ends[ends_len - 1] + count ;
                ends_len = ends_len + 1 ;
            }
        }
        
        __syncthreads();
    }
    __shared__ int depth ;
    __shared__ int start ;
    __shared__ int end ;
            
    if(idx == 0)
    {
        depth = ends_len - 1 ;
    }
    __syncthreads();
    while (depth > 0)
    {
        __syncthreads();
        if(idx==0)
        {
            start = ends[depth - 1 ] ;
            end = ends[depth] - 1 ;
        }
        __syncthreads();      
        if ( idx >= 0 && idx <= (end-start))
        {
            int w = S[start+idx] ;
            for(int r = graph->OA[w] ; r < graph->OA[w+1] ; r++)
            {
                int u = graph->CA[r] ;
                int wt_wu = graph->weight[r] ;
                if ( d[u] == d[w] + wt_wu )
                {
                    if (sigma[u] != 0)
                    {
                        atomicAdd(dependency + w, (sigma[w] * 1.0 / sigma[u]) * (1 + dependency[u]));
                    }
                }
            }
            if(w!=s)
            {
                atomicAdd(&BC[w],dependency[w]/2) ;
            }
        }
        if(idx == 0)
        {
            depth--;
        }
         __syncthreads() ;
    }
}

float *fun(Graph* h_graph)
{
    int node_count = h_graph->nodes;
    int edge_count = h_graph->edges;

    Graph *d_graph;  //DEVICE Graph

    cudaMalloc((void **)&d_graph, sizeof(Graph));

    //Copying graph from host to device but pointers have to be updated separately 
    //because pointers will contain address of host memory
    cudaMemcpy(d_graph, h_graph, sizeof(Graph), cudaMemcpyHostToDevice);  
 
    // Updating the address of OA(offset array) on the device graph
    int *d_OA;
    cudaMalloc((void **)&d_OA, sizeof(int) * (node_count + 1));
    cudaMemcpy(d_OA , h_graph->OA, sizeof(int) * (node_count + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_graph->OA), &d_OA, sizeof(int *), cudaMemcpyHostToDevice);

    // Updating the address of CA(Coordinates array) on the device graph
    int *d_CA;
    cudaMalloc((void **)&d_CA, sizeof(int) * (2 * edge_count + 1));
    cudaMemcpy(d_CA, h_graph->CA, sizeof(int) * (2 * edge_count + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_graph->CA), &d_CA, sizeof(int *), cudaMemcpyHostToDevice);


    // Updating the address of weight array on the device graph
    int *d_wt;
    cudaMalloc((void **)&d_wt, sizeof(int) * (2 * edge_count));
    cudaMemcpy(d_wt, h_graph->weight, sizeof(int) * (2 * edge_count), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_graph->weight), &d_wt, sizeof(int *), cudaMemcpyHostToDevice);
    
    float *bwCentrality = new float[node_count]();

    float *device_bwCentrality;
    cudaMalloc((void **)&device_bwCentrality, sizeof(float) * node_count);
    cudaMemcpy(device_bwCentrality, bwCentrality, sizeof(float) * node_count, cudaMemcpyHostToDevice);

    //TIMER
    cudaEvent_t device_start, device_end;
    cudaEventCreate(&device_start);
    cudaEventCreate(&device_end);
    cudaEventRecord(device_start);


    int *delta_v ;
    cudaMalloc((void **)&delta_v, sizeof(int) * node_count);
    cal_delta<<<1,node_count>>>(d_graph,delta_v);
    cudaDeviceSynchronize();

    kernel<<< node_count,node_count, 7*node_count*sizeof(int)+node_count*sizeof(float) >>> (d_graph ,device_bwCentrality ,delta_v , h_graph->nodes) ;
    cudaDeviceSynchronize();
    cudaMemcpy(bwCentrality,device_bwCentrality, sizeof(float) * node_count, cudaMemcpyDeviceToHost);


    cudaEventRecord(device_end);
    cudaEventSynchronize(device_end);
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    

    cudaFree(device_bwCentrality);
    cudaFree(delta_v);
    return bwCentrality;
}
int main(int argc, char *argv[])
{
    int m,n;
    int num1,num2;
    FILE *filePointer;
    char *filename = argv[1]; 
    //const char *filename = "ip1000.txt";
    filePointer = fopen( filename , "r") ; 
      
    //checking if file ptr is NULL
    if ( filePointer == NULL ) 
    {
        printf( "input.txt file failed to open." ) ; 
        return 0;
    }

    fscanf(filePointer, "%d", &n );     //scaning the number of vertices
    fscanf(filePointer, "%d", &m );     //scaning the number of edges

    Graph *graph = new Graph(n,m);   //HOST GRAPH

    vector <edgepairs> COO(2*m);
    int it=0;
    int wt ;
    for(int i=0 ; i<m ; i++ )  //scanning the edges
    {
        fscanf(filePointer, "%d", &num1) ;
        fscanf(filePointer, "%d", &num2) ;
        fscanf(filePointer, "%d", &wt) ;
        COO[it].x = num1 ;
        COO[it].y = num2 ;
        COO[it].wt = wt ;
        it++;
        COO[it].x = num2 ;
        COO[it].y = num1 ;
        COO[it].wt = wt ;
        it++;

    }
    // COO done...
    
    // sort the COO 
    sort(COO.begin(),COO.end(),compareTwoEdgePairs);
    
    for(int i=0;i<n+1;i++)
    {
        graph->OA[i] = 0;
    }
    graph->OA[0]=0;
    //initialize the Coordinates Array
    for(int i=0;i<2*m;i++)
    {
        graph->CA[i] = COO[i].y ;
        graph->weight[i] = COO[i].wt ;
    }
    //initialize the Offsets Array
    for(int i=0;i<2*m;i++)
    {
        graph->OA[COO[i].x + 1]++;     //store the frequency..
    }
    for(int i=0;i<n;i++)
    {
        graph->OA[i+1] += graph->OA[i];   // do cumulative sum..
    }
   
    float *bwC =  fun(graph);

    float maxBetweenness = -1;
    vector<int>indices;
    cout<<"Betweeness Centrality of all the nodes(vertices)\n";
    
    for (int i = 0; i < n; i++)
    {
        maxBetweenness = max(maxBetweenness, bwC[i]);
        cout<<"Node "<<i<<" : "<< bwC[i]<<endl;
    }
    for (int i = 0; i < n; i++)
    {
        if(maxBetweenness == bwC[i])
            indices.push_back(i);
    }

    cout << endl;

    cout<<"\nMaximum Betweenness Centrality = " << maxBetweenness<<endl;
    cout<<"Vertices with Maximum Betweenness Centrality: [";
    for(int i=0;i<indices.size();i++)
    {
        if(i != indices.size()-1)
            cout<<indices[i]<<" , ";
        else
            cout<<indices[i] ;
    }
    cout<<"]"<<endl;
    cout<<"Total device time taken : ";
    cout<<device_time_taken<<endl;
    delete[] bwC;
    delete graph ;
    
}
