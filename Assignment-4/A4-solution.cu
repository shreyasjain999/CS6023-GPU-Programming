#include<cuda.h>
#include<stdio.h>
#include<bits/stdc++.h>
using namespace std;
#define MAX 33
#define COLUMN 20

//number of pointers or number of child blocks [numberOfPointers = numberOfNodes + 1]
int numberOfPointers = 8;

struct Block{
    //number of nodes
    int tNodes;

    //for parent Block and index
    Block *parentBlock;
    
    //values
    int value[MAX];
    
    //child Blocks
    Block *childBlock[MAX];

    //record pointers
    int *recptr[MAX];

    Block(){    //constructor to initialize a block
        tNodes = 0;
        parentBlock = NULL;
        for(int i=0; i<MAX; i++){
            value[i] = INT_MAX;
            childBlock[i] = NULL;
            recptr[i] = NULL;
        }
    }

};

struct Range{
  int start;
  int end;   
};

struct Updatetuples{
  int key;
  int col;
  int uval;  
};

struct Node{
   int keys[MAX][COLUMN]; 
  int count;

    Node() {
        count = 0;
        for(int i=0 ; i<MAX; i++){
            for(int j=0 ; j<COLUMN ; j++){
            keys[i][j] = INT_MAX;            
            }
        }
    
    }

};

__host__ __device__ 
bool operator<(const Range &lhs, const Range &rhs) 
{
 return ( lhs.start < rhs.start ); 
 };

__host__ __device__ 
bool operator<(const Updatetuples &lhs, const Updatetuples &rhs) 
{
 return ( lhs.key < rhs.key ); 
 };

// creating GPU B+ structure here
vector <struct Node> key_region;
vector <int> child_prefix_sum;
int psum;                         
//end of GPU B+ structure


//declare root Block

Block *rootBlock = new Block();

//function to split the leaf nodes
void splitLeaf(Block *curBlock){
    int x, i, j;

    if(numberOfPointers%2)
        x = (numberOfPointers+1)/2;
    else x = numberOfPointers/2;

    Block *rightBlock = new Block();

    curBlock->tNodes = x;
    rightBlock->tNodes = numberOfPointers-x;
    rightBlock->parentBlock = curBlock->parentBlock;

    for(i=x, j=0; i<numberOfPointers; i++, j++){
        rightBlock->value[j] = curBlock->value[i];
        rightBlock->recptr[j] = curBlock->recptr[i];
        curBlock->value[i] = INT_MAX;
    }
    int val = rightBlock->value[0];
    int *rp = rightBlock->recptr[0];

    if(curBlock->parentBlock==NULL){
        Block *parentBlock = new Block();
        parentBlock->parentBlock = NULL;
        parentBlock->tNodes=1;
        parentBlock->value[0] = val;
        parentBlock->recptr[0] = rp;
        parentBlock->childBlock[0] = curBlock;
        parentBlock->childBlock[1] = rightBlock;
        curBlock->parentBlock = rightBlock->parentBlock = parentBlock;
        rootBlock = parentBlock;
        return;
    }
    else{   
        curBlock = curBlock->parentBlock;

        Block *newChildBlock = new Block();
        newChildBlock = rightBlock;

        for(i=0; i<=curBlock->tNodes; i++){
            if(val < curBlock->value[i]){
                swap(curBlock->value[i], val);
                curBlock->recptr[i] =  rp;
            }
        }

        curBlock->tNodes++;

        for(i=0; i<curBlock->tNodes; i++){
            if(newChildBlock->value[0] < curBlock->childBlock[i]->value[0]){
                swap(curBlock->childBlock[i], newChildBlock);
            }
        }
        curBlock->childBlock[i] = newChildBlock;

        for(i=0;curBlock->childBlock[i]!=NULL;i++){
            curBlock->childBlock[i]->parentBlock = curBlock;
        }
    }

}

//function to split the non leaf nodes
void splitNonLeaf(Block *curBlock){
    int x, i, j;

    x = numberOfPointers/2;

    Block *rightBlock = new Block();

    curBlock->tNodes = x;
    rightBlock->tNodes = numberOfPointers-x-1;
    rightBlock->parentBlock = curBlock->parentBlock;


    for(i=x, j=0; i<=numberOfPointers; i++, j++){
        rightBlock->value[j] = curBlock->value[i];
        rightBlock->recptr[j] = curBlock->recptr[i];
        rightBlock->childBlock[j] = curBlock->childBlock[i];
        curBlock->value[i] = INT_MAX;
        if(i!=x)curBlock->childBlock[i] = NULL;
    }

    int val = rightBlock->value[0];
    int *rp = rightBlock->recptr[0];
    memcpy(&rightBlock->value, &rightBlock->value[1], sizeof(int)*(rightBlock->tNodes+1));
    memcpy(&rightBlock->recptr, &rightBlock->recptr[1], sizeof(int *)*(rightBlock->tNodes+1));
    memcpy(&rightBlock->childBlock, &rightBlock->childBlock[1], sizeof(rootBlock)*(rightBlock->tNodes+1));

    for(i=0;curBlock->childBlock[i]!=NULL;i++){
        curBlock->childBlock[i]->parentBlock = curBlock;
    }
    for(i=0;rightBlock->childBlock[i]!=NULL;i++){
        rightBlock->childBlock[i]->parentBlock = rightBlock;
    }

    if(curBlock->parentBlock==NULL){
        Block *parentBlock = new Block();
        parentBlock->parentBlock = NULL;
        parentBlock->tNodes=1;
        parentBlock->value[0] = val;
        parentBlock->recptr[0] = rp;
        parentBlock->childBlock[0] = curBlock;
        parentBlock->childBlock[1] = rightBlock;

        curBlock->parentBlock = rightBlock->parentBlock = parentBlock;

        rootBlock = parentBlock;
        return;
    }
    else{   
        curBlock = curBlock->parentBlock;

        Block *newChildBlock = new Block();
        newChildBlock = rightBlock;

        for(i=0; i<=curBlock->tNodes; i++){
            if(val < curBlock->value[i]){
                swap(curBlock->value[i], val);
                curBlock->recptr[i] = rp ;
            }
        }

        curBlock->tNodes++;

        for(i=0; i<curBlock->tNodes; i++){
            if(newChildBlock->value[0] < curBlock->childBlock[i]->value[0]){
                swap(curBlock->childBlock[i], newChildBlock);
            }
        }
        curBlock->childBlock[i] = newChildBlock;

         for(i=0;curBlock->childBlock[i]!=NULL;i++){
            curBlock->childBlock[i]->parentBlock = curBlock;
        }
    }

}

void insertNode(Block *curBlock, int val, int *rp){

    for(int i=0; i<=curBlock->tNodes; i++){
        if(val < curBlock->value[i] && curBlock->childBlock[i]!=NULL){
            insertNode(curBlock->childBlock[i], val, rp);
            if(curBlock->tNodes==numberOfPointers)
                splitNonLeaf(curBlock);
            return;
        }
        else if(val < curBlock->value[i] && curBlock->childBlock[i]==NULL){
            swap(curBlock->value[i], val);
            curBlock->recptr[i] =  rp;
            if(i==curBlock->tNodes){
                    curBlock->tNodes++;
                    break;
            }
        }
    }

    if(curBlock->tNodes==numberOfPointers){

            splitLeaf(curBlock);
    }
}

void createGPUBplustree(vector < Block* > Blocks, int n){
    vector < Block* > newBlocks;

    for(int i=0; i<Blocks.size(); i++){ //for every block
        Block *curBlock = Blocks[i];
        struct Node t;
        t.count = curBlock->tNodes;
        for(int j =0 ; j<curBlock->tNodes ;j++ ){
          t.keys[j][0] = curBlock->value[j];
         
          for(int k=0; k < n-1 ; k++){
              t.keys[j][k+1] = 0;//*(curBlock->recptr[j] + k);
          }
           
        }

        key_region.push_back(t);
        
        int j;
        for(j=0; j<curBlock->tNodes; j++){ 

            if(curBlock->childBlock[j]!=NULL)
            {
            newBlocks.push_back(curBlock->childBlock[j]);
            psum++;
            }

            if(j==0){
                child_prefix_sum.push_back( psum );
            }
        
        }
        if(curBlock->value[j]==INT_MAX && curBlock->childBlock[j]!=NULL){
            newBlocks.push_back(curBlock->childBlock[j]);
            psum++;
        }
    }

    if(newBlocks.size()==0){ 
        Blocks.clear();
    }
    else {                 
        Blocks.clear();
        createGPUBplustree(newBlocks,n);
    }

}

//search code

__global__ void search( struct Node *a , int *b , int asize , int bsize , int *search_keys, int *mutex , int n , int *results){
    
    // task 1 is to assign individual searches to each threads
    int key = search_keys[blockIdx.x];
    __shared__ int index ;
    __shared__ int prev_index;

    index = 0;
    prev_index = 0;

    __syncthreads();
    
    
    // task 2 is to perform search using GPU B+ tree structure
    while(true){

        if( index > asize-1 ){
            break;
        }
        prev_index = index;

        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == key ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < key  && key < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > key ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < key ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }

        }
      bottom:  __syncthreads();
   }

 // store the indices after the search operation
  if(threadIdx.x == 0){
      index = prev_index;
      results[ blockIdx.x ] = index;
  }

   
}

//update code 
__global__ void update( struct Node *a , int *b , int asize , int bsize , struct Updatetuples *update_tuples , int *mutex ,int mode ){
    
    // task 1 is to assign individual updates to each threads
    struct Updatetuples tp = update_tuples[blockIdx.x];
    __shared__ int index ;
    __shared__ int prev_index;

    index = 0;
    prev_index = 0;

    __syncthreads();
    
    // task 2 is to perform search using GPU B+ tree structure
    while(true){
       
        if( index > asize-1 ){
            break;
        }
        prev_index = index;
        
        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == tp.key ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;   
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < tp.key  && tp.key < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > tp.key ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < tp.key ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }
           
        }
      bottom:  __syncthreads();
   }

    // task 3 is to add the uval to the given attr of searched tuple

    if( threadIdx.x == 0){
        index = prev_index;
        //int flag_found = 0;
    
        for(int i=0; i< a[index].count; i++){
          if( a[index].keys[i][0] == tp.key  ){
                //flag_found = 1;
                
                while( atomicCAS(mutex,0,1) != 0  );   //atomicity is needed while updation.
                a[index].keys[i][tp.col-1] = a[index].keys[i][tp.col-1] + tp.uval;
                atomicExch(mutex,0);
        }
    }
  }
	// else do nothing if the tuple with given key in not present..
   
    
}

//rangeQuery code 
__global__ void rangeQuery( struct Node *a , int *b , int asize , int bsize , struct Range *range_arr , int *mutex ,int n , int *results ){
    
    // task 1 is to assign individual search Ranges to each threads
    struct Range r = range_arr[blockIdx.x];
    __shared__ int index ;
    __shared__ int prev_index;

    index = 0;
    prev_index = 0;

    __syncthreads();
    
    
    // task 2 is to perform search using GPU B+ tree structure
    while(true){
       
        if( index > asize-1 ){
            break;
        }
        prev_index = index;
        
        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == r.start ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;   
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < r.start  && r.start < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > r.start ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < r.start ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }
           
        }
      bottom:  __syncthreads();
   }

  // store the indices for the searched values 
  if(threadIdx.x == 0){
      index = prev_index;
      results[ blockIdx.x ] = index;         
  }
  
}


__global__ void path_finder( struct Node *a , int *b , int asize , int bsize , int key, int n ){
    
    // task 1 is to assign individual searches to each threads
    __shared__ int index;

    index = 0;

    __syncthreads();
    
    
    // task 2 is to perform search using GPU B+ tree structure
    while(true){
       
        if( index > asize-1 ){
            break;
        }

	// print the first key in each node while traversing 
	if( threadIdx.x < a[index].count && threadIdx.x == 0 )
	printf("%d ", a[index].keys[0][0] );
        
        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == key ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;   
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < key  && key < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > key ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < key ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }
           
        }
      bottom:  __syncthreads();
   }

}

int main(int argc , char **argv){
    
    FILE *filePointer;
    char *filename = argv[1];
    char mode = *argv[2];
    filePointer = fopen( filename , "r") ; 
    
    if ( filePointer == NULL ) 
    {
        printf( "input.txt file failed to open." ) ; 
    }
    freopen("output.txt", "w", stdout);
	//printf("--1");
    vector < Block* > Blocks;

    int totalValues = 0;
    int m,n;
    
    fscanf(filePointer, "%d", &m );
    fscanf(filePointer, "%d", &n );

    int **database = (int **)malloc( (m) * sizeof(int *)); 
    for(int i=0; i< (m); i++) 
         database[i] = (int *)malloc(n * sizeof(int)); 


    //-------------------Initial Insertions form DB-----------------------
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            fscanf(filePointer, "%d" , &database[i][j] );
        }
    }

    // creating cpu b+ tree here
    for(int i=0;i<m;i++){
        
            insertNode(rootBlock, database[i][0] , &database[i][1] );
            totalValues++;
    }

            // create the GPU B+ structure
            key_region.clear();
            child_prefix_sum.clear();
            psum=0;
        
            Blocks.clear();
            Blocks.push_back(rootBlock);
            createGPUBplustree(Blocks,n);
            //puts("");
            for(int i=0 ; i<child_prefix_sum.size() ; i++  )
            {
                if( child_prefix_sum[i] == child_prefix_sum.size() - 1 )
                {
                    child_prefix_sum[i] = child_prefix_sum.size() ;
                }
            }
    //---------------------Initial Insertions end---------------------------

    int *mutex;
    struct Node *gpuA;
    int *gpuB;
    cudaMalloc(&mutex , sizeof(int) );
    cudaEvent_t start, stop;
    float milliseconds;

    int numofops;
    int ch;

    fscanf(filePointer, "%d" , &numofops );    

   for(int q=0; q < numofops ; q++ ){
        fscanf(filePointer, "%d" , &ch );
         
         if(ch == 1){
             //-----------------------------search op---------------------------------------
        int Ssize;
        fscanf(filePointer, "%d" , &Ssize );
        int *search_arr = (int *)malloc( Ssize*sizeof(int) );            
	int *result_arr = (int *)malloc( Ssize*sizeof(int)  );           
 
            for(int i=0 ; i< Ssize ; i++){
               fscanf(filePointer, "%d" , &search_arr[i] );  
            }   
           
            int *gpuC;
            int *results;

            cudaMemset(mutex,0,sizeof(int));

            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );
            cudaMalloc( &gpuC , Ssize*sizeof(int) );
	    cudaMalloc( &results , Ssize*sizeof(int) );

            cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuC , search_arr , Ssize*sizeof(int) , cudaMemcpyHostToDevice ); 

            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);    
            }
                
            search<<< Ssize, (numberOfPointers-1) >>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , gpuC , mutex , n , results );
            cudaDeviceSynchronize();

            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);
            }

	    cudaMemcpy( result_arr , results, Ssize*sizeof(int) , cudaMemcpyDeviceToHost );
	    int found=0;

	    for(int i=0;i<Ssize;i++){
		int index = result_arr[i];
		//printf("%d ",index);
		int key = search_arr[i];
		found=0;
		for(int j=0;j<key_region[index].count;j++){
		
			if( key_region[index].keys[j][0] == key  ){
                		found = 1;
				for(int k=0 ; k < n ; k++)
                		printf("%d " , key_region[index].keys[j][k]  );
        		}
		}
		if(!found){
			printf("-1");
		}
            printf("\n");
            }


            //------------------------search end--------------------------------------------

         }

        else if(ch == 2)
        {
            //--------------------------Range Query----------------------------------------
            
            int Rsize;
            fscanf(filePointer, "%d" , &Rsize );             
            //Range range_arr[Rsize];
            Range *range_arr = (Range *)malloc( Rsize*sizeof(Range) );
	    int *result_arr = (int *)malloc( Rsize*sizeof(int)  );        
 
            for(int i=0 ; i< Rsize ; i++){
               fscanf(filePointer, "%d" , &range_arr[i].start );
               fscanf(filePointer, "%d" , &range_arr[i].end );
            }
         
            //thrust::sort(range_arr, range_arr + Rsize );
         
            struct Range *gpuD;
            int *results;
            cudaMemset(mutex,0,sizeof(int));

            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );
            cudaMalloc( &gpuD, Rsize*sizeof(struct Range) );
	    cudaMalloc( &results , Rsize*sizeof(int) );

            cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);
            cudaMemcpy(gpuD , range_arr , Rsize*sizeof(struct Range) , cudaMemcpyHostToDevice );

            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);    
            }
            
            rangeQuery<<<Rsize,(numberOfPointers-1)>>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , gpuD , mutex , n ,results );
            cudaDeviceSynchronize();
         
            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);    
            }

	    cudaMemcpy( result_arr , results, Rsize*sizeof(int) , cudaMemcpyDeviceToHost );
            int found=0;
	    for(int itr=0;itr<Rsize;itr++){
		found=0;
		int i = result_arr[itr];
		int j=0;
		//printf("%d ",i);		
		while(true){
     
        		if( key_region[i].keys[j][0] > range_arr[itr].end )
          			break;     

      		  	if( i > key_region.size() - 1 )
          			break;

        		if(  range_arr[itr].start <= key_region[i].keys[j][0] && key_region[i].keys[j][0] <= range_arr[itr].end ){
              			//printf("\n");
				found = 1;
                 		for(int k=0 ; k < n ; k++){
                  			printf("%d " , key_region[i].keys[j][k] );   
                 		}
                		printf("\n");      
        		}
        		j++;

        		if( j >= key_region[i].count ){
        		i++;
        		j=0;   
        		}
    		}
		if(!found)
			printf("-1\n");
            }
            
            //-------------------------Range Query end-------------------------------------

        }

        else if( ch == 3)
        {
        //------------------------add operation-------------------------------------

            int Usize;
            fscanf(filePointer, "%d" , &Usize ); 
            //Updatetuples tp[Usize]; 
            Updatetuples *tp = (Updatetuples *)malloc( Usize*sizeof(Updatetuples) );

            for(int i=0 ; i< Usize ; i++){
               fscanf(filePointer, "%d" , &tp[i].key );
               fscanf(filePointer, "%d" , &tp[i].col );
               fscanf(filePointer, "%d" , &tp[i].uval );  
            }
         
            //thrust::sort(tp, tp + Usize );
            
            Updatetuples *gpuE;
            cudaMemset(mutex,0,sizeof(int));

            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );
            cudaMalloc( &gpuE, Usize*sizeof(struct Updatetuples) );
            
            cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuE , tp , Usize*sizeof(struct Updatetuples) , cudaMemcpyHostToDevice );
            
            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
            }
    
            update<<<Usize,(numberOfPointers-1)>>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , gpuE ,mutex, mode );
            cudaMemcpy( &key_region[0], gpuA , key_region.size() * sizeof(struct Node) , cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
         
            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);
            }
            

        //------------------------add operation end---------------------------------
        }

        else if(ch == 4)
        {
         //-------------------------Path printing opertions----------------------------------

            //int Psize;
            //fscanf(filePointer, "%d" , &Psize ); 

	    int key;
	    fscanf(filePointer, "%d" , &key );            
         
            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );

	    cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);	

	    if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
            }

	    path_finder<<< 1, (numberOfPointers-1) >>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , key , n  );
            cudaDeviceSynchronize();

            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);
            }
	    printf("\n");            
            //-------------------------Path printing End----------------------------------------
        
        }
    
    }
       
      return 0;
}

