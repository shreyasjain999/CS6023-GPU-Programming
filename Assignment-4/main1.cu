
#include<stdio.h>
#include<cuda.h>
#include<string.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

int null = INT_MAX;
int ninf = INT_MIN;


typedef struct Node{
    Node* parent;
    thrust :: host_vector<int> keys;
    thrust :: host_vector<Node*> pointer;
    bool isLeaf;
    bool isDead;
    Node* buffer;

}Node;

typedef struct dNode{
    int keys[7];
    Node* pointer[8];
    bool isLeaf;
    int no_keys=0;
    int *data_pointer[7];
}dNode;





__global__ void range(int *prefix_sum , dNode *nodes,int **result,int* count ,int n , int d , int tree_size , int *ab)
{
    __shared__ int node_idx;
    node_idx=0;
    int idx = threadIdx.x ;
    bool flag = true;
    __shared__ int a;
    __shared__ int b;
     dNode curr;
    __shared__ int c;
    c=0;
    __shared__ int level;
    level=0;
    a=ab[blockIdx.x*2 + 0];
    b=ab[blockIdx.x*2 + 1];
    __syncthreads();
    if(a!=-1 && b!=-1)
    {
          
          while(true)
          {
              curr = nodes[node_idx];
              if(level >= n || node_idx>=tree_size)
              {
                break;
              }
              if(curr.isLeaf)
                {
                  flag=true;
                  break;
                }
                int diff=INT_MAX;
                __shared__ int min_idx,min_diff;
                min_idx=0;
                min_diff=INT_MAX;
                if(idx < curr.no_keys )
                      diff = abs(a - curr.keys[idx]);
                __syncthreads();
                atomicMin(&min_diff,diff);
              //printf("min_diff : %d\n",min_diff);
                if(min_diff == diff)
                {
                      min_idx = idx ;
                    if(min_idx == 0 )
                    {
                          if(a<curr.keys[0])
                          {
                             
                              node_idx = prefix_sum[node_idx] ;
                    
                        }
                          else
                                node_idx = prefix_sum[node_idx]+1;
                    }
                    else if(min_idx == d-1 )
                    {
                          if(a<curr.keys[d-1])
                              node_idx = prefix_sum[node_idx] + d-1;
                          else
                              node_idx = prefix_sum[node_idx] + d   ;
                    }
                    else
                    {
                          if(a<curr.keys[min_idx])
                          {
                                node_idx = prefix_sum[node_idx] + min_idx;
                          }
                          else if(a>=curr.keys[min_idx])
                          {                   
                              node_idx = prefix_sum[node_idx] + min_idx + 1;                
                          }
                    }
                } 
          }
    __shared__ int ele_count;
    ele_count=0;
    bool flag=false;
    int ite=node_idx;
    __syncthreads();
    while(ele_count < n && idx==0)
    {
        for(int i=0;i<curr.no_keys;i++)
        {
            
            if( a <= curr.keys[i] && curr.keys[i] <= b )
            {
                c++;
                result[blockIdx.x * n + ele_count]=curr.data_pointer[i];
                ele_count++;
            }
            else if (curr.keys[i] > b)
                flag=true;

        }
        if(flag)
          break;
        ite++;
        if(ite >= tree_size)
          break;
        curr = nodes[ite];
        
        
    }
    // printf("Block %d : %d\n",blockIdx.x,c);
    count[blockIdx.x] = c;
    //printf("Block %d : %d\n",blockIdx.x,c);
    //printf("%d ::: %d\n",blockIdx.x, count[blockIdx.x]);
}
}
__global__ void find(int *prefix_sum , dNode *nodes , int **result , int* found , int tree_size , int n , int d , int *keys)
{
    int idx = threadIdx.x ;
    __shared__ int node_idx;
    node_idx=0;
    __shared__ bool flag;
    flag=false;
    dNode curr;
    __shared__ int key;
    key=keys[blockIdx.x];
    __shared__ int level;
    level=0;
    __syncthreads();
    if(key!=-1)
    {
      while(true)
      {
        //printf("\n");
        
        level++;
        if(level >= n || node_idx>=tree_size)
        {
            break;
        }
        curr = nodes[node_idx];
        if(curr.isLeaf )
        {
            flag=true;
            break;
        }
        int diff=INT_MAX;
        __shared__ int min_idx,min_diff;
        min_idx=0;
        min_diff=INT_MAX;
        if(idx < curr.no_keys )
            diff = abs(key - curr.keys[idx]);
        atomicMin(&min_diff,diff);
        __syncthreads();
        //printf("min_diff : %d\n",min_diff);
        if(min_diff == diff)
        {
            min_idx = idx ;
            if(min_idx == 0 )
            {
                if(key<curr.keys[0])
                {
                    node_idx = prefix_sum[node_idx] ;
                }
                else
                    node_idx = prefix_sum[node_idx]+1;
            }
            else if(min_idx == d-1 )
            {
                if(key<curr.keys[d-1])
                    node_idx = prefix_sum[node_idx] + d-1;
                else
                    node_idx = prefix_sum[node_idx] + d ;
            }
            else
            {
                if(key<curr.keys[min_idx])
                {
                    node_idx = prefix_sum[node_idx] + min_idx;
                }
                else if(key>=curr.keys[min_idx])
                {
                    node_idx = prefix_sum[node_idx] + min_idx + 1;
                }
            }
        }
      __syncthreads();
      }
    }
    
    if(flag)
    {
        if(curr.keys[idx] == key)
        {
          //printf("FOUND THE KEY : %d.\n",key);

          found[blockIdx.x] = 1;
          result[blockIdx.x] = curr.data_pointer[idx];
          //printf("AAAA\n");
        }    
    }

}
__global__ void path_trace(int *prefix_sum , dNode *nodes,int* keys, int *count , int tree_size , int n , int d,int k)
{
    int idx = threadIdx.x ;
    __shared__ int node_idx;
    node_idx=0;
    __shared__ int it;
    it=0;
    bool flag=false;
    dNode curr;
    __shared__ int key;
    key=k;
    __shared__ int level;
    level = 0;
    while(true)
    {
        //printf("\n");
        level++;
        if(level >= n || node_idx>=tree_size)
        {
            break;
        }
        curr = nodes[node_idx];
        if(idx==0)
        {
            keys[it]=curr.keys[0];
            it++;
            ++*count;
        }
        if(curr.isLeaf )
        {
            flag=true;
            break;
        }
        int diff=INT_MAX;
        __shared__ int min_idx,min_diff;
        min_idx=0;
        min_diff=INT_MAX;
        if(idx < curr.no_keys )
            diff = abs(key - curr.keys[idx]);
        atomicMin(&min_diff,diff);
        __syncthreads();
        //printf("min_diff : %d\n",min_diff);
        if(min_diff == diff)
        {
            min_idx = idx ;
            if(min_idx == 0 )
            {
                if(key<curr.keys[0])
                {
                    node_idx = prefix_sum[node_idx] ;
                }
                else
                    node_idx = prefix_sum[node_idx]+1;
            }
            else if(min_idx == d-1 )
            {
                if(key<curr.keys[d-1])
                    node_idx = prefix_sum[node_idx] + d-1;
                else
                    node_idx = prefix_sum[node_idx] + d ;
            }
            else
            {
                if(key<curr.keys[min_idx])
                {
                    node_idx = prefix_sum[node_idx] + min_idx;
                }
                else if(key>=curr.keys[min_idx])
                {
                    node_idx = prefix_sum[node_idx] + min_idx + 1;
                }
            }
        }
      __syncthreads();
    }
}


Node* init_node(int n, bool flag)
{
    Node* node = new Node;
    node->parent = NULL;
    node->keys = thrust :: host_vector<int>(n, null);
    node->pointer = thrust :: host_vector<Node*>(n+1);
    node->isLeaf = flag;
    node->isDead = false;
    node->buffer = NULL;
    return node;
}

void unMark(Node* parent, Node* child, int value)
{
    if(parent != NULL)
    {
        bool flag = false;
        for (int i = 1; i < parent->pointer.size(); ++i)
        {
            if(parent->pointer[i] == child)
            {
                flag = true;
                parent->keys[i - 1] = value;
            }
        }
        if(parent->isDead && flag)
            unMark(parent->parent, parent, value);
    }
}

Node* insert(Node* node, int value)
{
    Node* root = NULL;
    int node_size = node->keys.size();
    bool full_flag = false;
    if(node->keys[node_size - 1] != null)
        full_flag = true;
    if(full_flag)
    {
        thrust :: host_vector<int> tempKeys = node->keys;
        thrust :: host_vector<Node*> tempPointers = node->pointer;
        int tempIndex = thrust :: upper_bound(tempKeys.begin(), tempKeys.end(), value) - tempKeys.begin();
        int ubp, newVal;
        tempKeys.insert(tempKeys.begin() + tempIndex, value);
      
        if(!node->isLeaf)
            tempPointers.insert(tempPointers.begin() + tempIndex + 1, node->buffer);
        Node* new_node = init_node(node_size, node->isLeaf);
        new_node->parent = node->parent;
  
        if(node->isLeaf)
        {
            new_node->pointer[node_size] = node->pointer[node_size];
            node->pointer[node_size] = new_node;

            double tempFloat = node_size + 1;
            if(node_size % 2 == 1)
                ubp = (int)ceil(tempFloat/2);
            else
                ubp = (int)ceil(tempFloat/2)-1;
        }
        else
        {
            double tempFloat = node_size + 2;
            if(node_size % 2 == 1)
                ubp = (int)ceil((tempFloat)/2);
            else
                ubp = (int)ceil(tempFloat/2)-1;
            for (int i = 0; i < tempPointers.size(); ++i)
            {
                if(i <= ubp)
                    node->pointer[i] = tempPointers[i];
                else
                {
                    new_node->pointer[i - ubp-1] = tempPointers[i];
                    new_node->pointer[i - ubp-1]->parent = new_node;
                    if(i <= node_size)
                        node->pointer[i] = NULL;
                }
            }
              newVal = tempKeys[ubp];
              tempKeys.erase(tempKeys.begin() + ubp);
          }
          for (int i = 0; i < tempKeys.size(); ++i)
          {
                if(i < ubp)
                      node->keys[i] = tempKeys[i];
                else
                {
                      new_node->keys[i - ubp] = tempKeys[i];
                      if(i < node_size)
                            node->keys[i] = null;
                }
          }

          if(node->isDead && value != node->keys[0] && tempIndex < ubp)
          {
                node->isDead = false;
                unMark(node->parent, node, value);
          }

          tempIndex = upper_bound(new_node->keys.begin(), new_node->keys.end(), node->keys[ubp - 1]) - new_node->keys.begin();
        
        if(new_node->keys[tempIndex] == null)
          {
                newVal = new_node->keys[0];
                new_node->isDead = true;
          }
          else if(node->isLeaf)
                newVal = new_node->keys[tempIndex];

    
          if(node->parent != NULL)
          {
              
                node->parent->buffer = new_node;
                root = insert(node->parent, newVal);
          }
          else
          {
                root = init_node(node_size, false);
                root->keys[0] = newVal;
                root->pointer[0] = node;
                root->pointer[1] = new_node;
                node->parent = root;
                new_node->parent = root;
          }   
    }
    else
    {
        bool insert_flag = false;
        int tempKey = null;
        Node* tempPointer = NULL;
        for (int i = 0; i < node_size; i++)
        {
              if(insert_flag)
              {
                    int temp = node->keys[i] ; 
                    node->keys[i]=tempKey ;
                    tempKey = temp ;
                    if(!node->isLeaf)
                    {  
                        Node* temp = node->pointer[i + 1];
                        node->pointer[i + 1] = tempPointer ;
                        tempPointer = temp;
                        //swap(node->pointer[i + 1], tempPointer);
                    }
            }
            else
            {
                  if(value < node->keys[i] || node->keys[i] == null)
                  {
                        insert_flag = true;
                        tempKey = node->keys[i];
                        node->keys[i] = value;
                        if(!node->isLeaf)
                        {
                              tempPointer = node->pointer[i + 1];
                              node->pointer[i + 1] = node->buffer;
                        }
                }
                if(value != node->keys[0] && node->isDead)
                {
                      node->isDead = false;
                      unMark(node->parent, node, value);
                }
            }
        }
    }
    return root;
}

Node* find_pos(Node* node, int value, bool up)
{
    while(!node->isLeaf)
    {
        int lb = ninf, ub, node_size = node->keys.size(), index;
        for (int i = 0; i < node_size; i++)
        {
            if(node->keys[i] == null)
            {
                index = i;
                break;
            }
            ub = node->keys[i];
            if(lb <= value && value < ub)
            {
                index = i;
                break;
            }
            else if(lb <= value && value == ub && !up && node->pointer[i + 1]->isDead)
            {
                index = i;
                break;
            }
            else
                index = i + 1;
            lb = ub;
        }
        node = node->pointer[index];
    }
    return node;
}
Node* insert_(Node* root, int value)
{
    Node* temp = root;
    temp = insert(find_pos(root, value, true), value);
    if(temp != NULL)
        root = temp;
    return root;
}



int main(int argc,char **argv)
{
    
    int n,m;

    FILE *inputfilepointer;

    char *inputfilename = argv[1];
    

    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &n );      
    fscanf( inputfilepointer, "%d", &m );      

    int arr[n][m];
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            fscanf( inputfilepointer, "%d", &arr[i][j] );
        }
    }
    int d=7;
    int keys[n];
    int min_key = INT_MAX;
    for(int i=0;i<n;i++)
    {
        keys[i]=arr[i][0];
        if(min_key > keys[i])
            min_key=keys[i];
    }
    Node *root=init_node(d,true);
    for(int i=0;i<n;i++)
    {   
        root = insert_(root , keys[i]);
    }
    int idx = 0 ;
    thrust :: host_vector<int>t;
    Node *node = root ;
    thrust :: host_vector<Node*>tree;
    tree.push_back(node);
    t.push_back(1);
    while (idx < tree.size())
    {   
        int count=0;
        Node *temp = tree[idx];
        idx++;
        if(!temp->isLeaf)
        {
            for(int i=0;i<=d;i++)
            {
                if(temp->pointer[i] != NULL)
                {
                    count++;
                    tree.push_back(temp->pointer[i]);
                }
            }   
            t.push_back(count);
        }
    
    }
 
    dNode* dtree=(dNode*)malloc(tree.size()*sizeof(dNode));
    for(int i=0;i<tree.size();i++)
    {
        Node *curr=tree[i];
        dNode new_curr;
        new_curr.isLeaf = curr->isLeaf;
        for(int j=0;j<d;j++)
        {
              new_curr.keys[j] = curr->keys[j];
            new_curr.pointer[j] = curr->pointer[j];
        }
        new_curr.pointer[d]=curr->pointer[d];
        dtree[i]=new_curr;
    }
 
    for(int i=0;i<tree.size();i++)
    {
        int count=0;
        Node* curr = tree[i];
        for(int j=0;j<d;j++)
        {
            if(curr->keys[j]!=null )
              count++;
        }
        dtree[i].no_keys=count;
        if(curr->isLeaf)
        {
            for(int j=0;j<dtree[i].no_keys;j++)
            {
                int val = curr->keys[j];
                for(int k=0;k<n;k++)
                {
                    if(val == arr[k][0])
                    {
                        dtree[i].data_pointer[j]=&arr[k][0];
                        break;
                    }
                }
            }
        }
       
    }
 
 
    
    
    int prefix_sum[t.size()-1];
    prefix_sum[0]=1;
    for(int i=1;i<t.size()-1;i++)
    {
        prefix_sum[i]=t[i]+prefix_sum[i-1];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    dNode* d_tree ;
    cudaMalloc(&d_tree , tree.size()*sizeof(dNode)) ;
    cudaMemcpy(d_tree , dtree, tree.size()*sizeof(dNode), cudaMemcpyHostToDevice);
    
    int * d_prefix_sum ;
    cudaMalloc(&d_prefix_sum,(t.size()-1)*sizeof(int));
    cudaMemcpy(d_prefix_sum , prefix_sum, (t.size()-1)*sizeof(int), cudaMemcpyHostToDevice);

    char *outputfilename = argv[2];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");


    int q;
    //scanf("%d",&q);
    fscanf( inputfilepointer, "%d", &q );      

    while(q--)
    {
        
        int type;
        fscanf( inputfilepointer, "%d", &type );          
        
        if(type == 1)
        {
           
            int p;
            
            fscanf( inputfilepointer, "%d", &p );    
            int find_keys[p];
            for(int i=0;i<p;i++)
            {
                fscanf( inputfilepointer, "%d", &find_keys[i] );      
               
            }
            
            int no_calls=ceil(float(p)/float(100));
            int extra = p%100;
            int idx=0;
            int *h_result[100];

            if(extra == 0)
            {
                for(int i=0;i<(no_calls)*100;i+=100)
                { 
                    idx=i;
                    int h_keys[100];
                    int ite=0;
                    for(int x=i;x<i+100;x++)
                    {
                      h_keys[ite]=find_keys[x];
                      ite++;
                    }
                    int *d_keys;
                    cudaMalloc(&d_keys,100*sizeof(int));
                    cudaMemcpy(d_keys,h_keys,100*sizeof(int),cudaMemcpyHostToDevice);

                    int found[100];
                    for(int y=0;y<100;y++)
                      found[y]=0;
                    int *d_found;
                    cudaMalloc(&d_found , 100*sizeof(int));
                    cudaMemcpy(d_found , found, 100*sizeof(int) , cudaMemcpyHostToDevice);
                    int **d_result;
                    cudaMalloc(&d_result,100*sizeof(int*));
                    find<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_found  , tree.size() , n , d , d_keys);
                    cudaMemcpy(h_result,d_result,100*sizeof(int*),cudaMemcpyDeviceToHost);
                    cudaMemcpy(found , d_found , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    for(int j=0;j<100;j++)
                    {
                        if(found[j])
                        {
                            int * addr = h_result[j];
                            for(int k=0;k<m;k++)
                            {
                                
                                fprintf( outputfilepointer, "%d ", addr[k]);
                            }
                          
                            fprintf( outputfilepointer, "\n");
                        }
                        else
                        {
                            fprintf( outputfilepointer, "-1\n");
                            
                        }
                    }
                }
            }
            if(extra!=0)
            {
                for(int i=0;i<(no_calls-1)*100;i+=100)
                { 
                    idx=i;
                    int h_keys[100];
                    int ite=0;
                    for(int x=i;x<i+100;x++)
                    {
                      h_keys[ite]=find_keys[x];
                      ite++;
                    }
                    int *d_keys;
                    cudaMalloc(&d_keys,100*sizeof(int));
                    cudaMemcpy(d_keys,h_keys,100*sizeof(int),cudaMemcpyHostToDevice);
                    int found[100];
                    for(int y=0;y<100;y++)
                      found[y]=0;
                    int *d_found;
                    cudaMalloc(&d_found , 100*sizeof(int));
                    cudaMemcpy(d_found , found, 100*sizeof(int) , cudaMemcpyHostToDevice);
                    int **d_result;
                    cudaMalloc(&d_result,100*sizeof(int*));
                    find<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_found  , tree.size() , n , d , d_keys);
                    cudaMemcpy(h_result,d_result,100*sizeof(int*),cudaMemcpyDeviceToHost);
                    cudaMemcpy(found , d_found , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    for(int j=0;j<100;j++)
                    {
                        if(found[j])
                        {
                            int * addr = h_result[j];
                            for(int k=0;k<m;k++)
                            {
                                fprintf( outputfilepointer, "%d ", addr[k]);
                                
                            }
                            
                            fprintf( outputfilepointer, "\n");
                        }
                        else
                        {
                            fprintf( outputfilepointer,"-1\n" );
                        }
                    }
                }
                int h_keys[100]={-1};
                idx=0;
                for(int i=(no_calls-1)*100;i<p;i++)
                {  
                    h_keys[idx]=find_keys[i];
                    idx++;
                }
                int *d_keys;
                cudaMalloc(&d_keys,100*sizeof(int));
                cudaMemcpy(d_keys,h_keys,100*sizeof(int),cudaMemcpyHostToDevice);

                int found[100];
                for(int y=0;y<100;y++)
                      found[y]=0;
                int *d_found;
                cudaMalloc(&d_found , 100*sizeof(int));  
                cudaMemcpy(d_found , found, 100*sizeof(int) , cudaMemcpyHostToDevice);
        
                int **d_result;
        
                cudaMalloc(&d_result,100*sizeof(int*));
        
                find<<<100,7>>>(d_prefix_sum , d_tree , d_result  , d_found ,  tree.size() , n , d , d_keys);
        
                cudaMemcpy(h_result,d_result , 100*sizeof(int*) ,cudaMemcpyDeviceToHost);
                cudaMemcpy(found , d_found , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
      
                for(int i=0;i<extra;i++)
                {
                    if(found[i])
                    {
                        int * addr = h_result[i];
                        for(int k=0;k<m;k++)
                        {
                            fprintf( outputfilepointer, "%d ", addr[k]);
                            //printf("%d ",addr[k]);
                        }
                        //printf("\n");
                        fprintf( outputfilepointer, "\n");
                    }
                    else
                    {
                        //printf("-1\n");
                        fprintf( outputfilepointer, "-1\n");
                    }
                }
            }
        
        
        }
        else if(type == 2)
        {
         

          int p;
          fscanf( inputfilepointer, "%d", &p );     
          int points[p][2];
          for(int i=0;i<p;i++)
          {
                fscanf( inputfilepointer, "%d", &points[i][0] );      //scaning for toll tax zone passing time
                fscanf( inputfilepointer, "%d", &points[i][1] );      //scaning for toll tax zone passing time
          }


          int no_calls=ceil(float(p)/float(100));
          int extra = p%100;
          if(extra == 0)
          {
            for(int i=0;i<(no_calls)*100;i+=100)
            { 
                idx=0;
                int ab[100][2];
                for(int x=i;x<i+100;x++)
                {
                  ab[idx][0]=points[x][0];
                  ab[idx][1]=points[x][1];
                  idx++;
                }
                
                int *d_ab;
                cudaMalloc(&d_ab,200*sizeof(int));
                cudaMemcpy(d_ab,ab,200*sizeof(int),cudaMemcpyHostToDevice);

                int **h_result;
                h_result = (int**)malloc(100*n*sizeof(int*));   
                int **d_result;
                cudaMalloc(&d_result,100*n*sizeof(int*));
               
             
                int count[100];
                for(int y=0;y<100;y++)
                      count[y]=-1;
                int *d_count;
                cudaMalloc(&d_count,100*sizeof(int));
                cudaMemcpy(d_count ,count , 100*(sizeof(int)) , cudaMemcpyHostToDevice);
                range<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_count ,  n , d , tree.size(), d_ab);
                cudaMemcpy(h_result,d_result,100*n*sizeof(int*),cudaMemcpyDeviceToHost);
                cudaMemcpy(count , d_count , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                for(int l=0;l<100;l++)
                { 
                    if(count[l] > 0)
                    {
                      for(int j=0;j<count[l];j++)
                      { 
                        int *addr =   h_result[l*n + j];
                          for(int k=0;k<m;k++)
                          {
                              fprintf( outputfilepointer, "%d " , addr[k]);
                              //printf("%d ",addr[k]);
                          }
                          //printf("\n");
                          fprintf( outputfilepointer, "\n");
                      }
                    }
                    else if(count[l]==0)
                    {
                        fprintf( outputfilepointer, "-1\n" );
                        //printf("-1\n");
                    }
                }
                
            }
        }
       if(extra!=0)
        {
            for(int i=0;i<(no_calls-1)*100;i+=100)
            { 
                idx=0;
                int ab[100][2];
                for(int x=i;x<i+100;x++)
                {
                  printf("%d & %d \n", idx , x);
                  ab[idx][0]=points[x][0];
                  ab[idx][1]=points[x][1];
                  idx++;
                }
                
                int *d_ab;
                cudaMalloc(&d_ab,200*sizeof(int));
                cudaMemcpy(d_ab,ab,200*sizeof(int),cudaMemcpyHostToDevice);

                int **h_result;
                h_result = (int**)malloc(100*n*sizeof(int*));                
                int **d_result;
                cudaMalloc(&d_result,100*n*sizeof(int*));

                
                int count[100];
                for(int y=0;y<100;y++)
                    count[y]=-1;
                
                int *d_count;
                cudaMalloc(&d_count,100*sizeof(int));
                cudaMemcpy(d_count ,count , 100*(sizeof(int)) , cudaMemcpyHostToDevice);
                range<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_count ,  n , d , tree.size(), d_ab);
                cudaMemcpy(h_result,d_result,100*n*sizeof(int*),cudaMemcpyDeviceToHost);
                cudaMemcpy(count , d_count , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                
                for(int l=0;l<100;l++)
                { 
                    if(count[l] > 0)
                    {
                      for(int j=0;j<count[l];j++)
                      { 
                        int *addr =  h_result[l*n + j];
                          for(int k=0;k<m;k++)
                          {
                              fprintf( outputfilepointer, "%d " , addr[k]);
                              //printf("%d ",addr[k]);
                          }
                          //printf("\n");
                          fprintf( outputfilepointer, "\n");
                      }
                    }
                    else if(count[l]==0)
                    {
                        fprintf( outputfilepointer, "-1\n" );
                        //printf("-1\n");
                    }
                }
                
            }
            int ab[100][2];
            for(int x=0;x<100;x++)
            {
               ab[x][0]=-1;
               ab[x][1]=-1;
            }
            idx=0;
            for(int i=(no_calls-1)*100;i<p;i++)
            {  
                ab[idx][0]=points[i][0];
                ab[idx][1]=points[i][1];
                idx++;
            }
            int *d_ab;
            cudaMalloc(&d_ab,200*sizeof(int));
            cudaMemcpy(d_ab,ab,200*sizeof(int),cudaMemcpyHostToDevice);
            
        
            
            int **h_result;
            h_result = (int**)malloc(100*n*sizeof(int*));

         
            int **d_result;
           
            cudaMalloc(&d_result,100*n*sizeof(int*));
           
            int count[100];
            for(int y=0;y<100;y++)
                  count[y]=-1;

                int *d_count;
                cudaMalloc(&d_count,100*sizeof(int));
                cudaMemcpy(d_count ,count , 100*(sizeof(int)) , cudaMemcpyHostToDevice);
               
                range<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_count ,  n , d , tree.size(), d_ab);
                cudaDeviceSynchronize();
                cudaMemcpy(h_result,d_result,100*n*sizeof(int*),cudaMemcpyDeviceToHost);
                cudaMemcpy(count , d_count , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
               
                for(int l=0;l<extra;l++)
                { 
                    if(count[l] > 0)
                    {
                      for(int j=0;j<count[l];j++)
                      { 
                        int *addr = h_result[l*n + j];
                          for(int k=0;k<m;k++)
                          {
                              fprintf( outputfilepointer, "%d " , addr[k]);
                              //printf("%d ",addr[k]);
                          }
                          //printf("\n");
                          fprintf( outputfilepointer, "\n");
                      }
                    }
                    else if(count[l]==0)
                    {
                        fprintf( outputfilepointer, "-1\n" );
                        //printf("-1\n");
                    }
                }
            
            
          }

          
          
        }
        else if(type == 3)
        {
            //int p=3;
            int p;
            //scanf("%d",&p);
            fscanf( inputfilepointer, "%d", &p );      //scaning for toll tax zone passing time

            int find_keys[p][3];
            //int find_keys[p][3]={{21,4,987},{18,3,143},{6,2,100}};
            for(int i=0;i<p;i++)
            {
                //scanf("%d",&find_keys[i][0]);
                //scanf("%d",&find_keys[i][1]);
                //scanf("%d",&find_keys[i][2]);
                fscanf( inputfilepointer, "%d", &find_keys[i][0] );      //scaning for toll tax zone passing time
                fscanf( inputfilepointer, "%d", &find_keys[i][1] );      //scaning for toll tax zone passing time
                fscanf( inputfilepointer, "%d", &find_keys[i][2] );      //scaning for toll tax zone passing time
            }
            int no_calls=ceil(float(p)/float(100));
            int extra = p%100;
            int idx=0;
            if(extra == 0)
            {
                for(int i=0;i<(no_calls)*100;i+=100)
                { 

                    idx=i;
                    int h_keys[100];
                    int ite=0;
                    for(int x=i;x<i+100;x++)
                    {
                      h_keys[ite]=find_keys[x][0];
                      ite++;
                    }
                    int *d_keys;
                    cudaMalloc(&d_keys,100*sizeof(int));
                    cudaMemcpy(d_keys,h_keys,100*sizeof(int),cudaMemcpyHostToDevice);

                    int found[100];
                    for(int y=0;y<100;y++)
                      found[y]=0;
                    int *d_found;
                    cudaMalloc(&d_found , 100*sizeof(int));
                    cudaMemcpy(d_found , found, 100*sizeof(int) , cudaMemcpyHostToDevice);

                    int *h_result[100];
                    int **d_result;
                    cudaMalloc(&d_result,100*sizeof(int*));
                    find<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_found  , tree.size(), n , d , d_keys);
                    cudaDeviceSynchronize();
                    cudaMemcpy(h_result,d_result,100*sizeof(int*),cudaMemcpyDeviceToHost);
                    cudaMemcpy(found , d_found , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                    
                    for(int j=0;j<100;j++)
                    {
                        if(found[j])
                        {
                            int * addr = h_result[j];
                            addr[find_keys[i+j][1]-1] = addr[find_keys[i+j][1]-1] + find_keys[i+j][2];
                        }
                        
                    }
                }
            }
            if(extra!=0)
            {
                for(int i=0;i<(no_calls-1)*100;i+=100)
                { 
                    //printf("Inside type 3 : %d\n",i);
                    idx=i;
                    int h_keys[100];
                    int ite=0;
                    for(int x=i;x<i+100;x++)
                    {
                      h_keys[ite]=find_keys[x][0];
                      ite++;
                    }
                    int *d_keys;
                    cudaMalloc(&d_keys,100*sizeof(int));
                    cudaMemcpy(d_keys,h_keys,100*sizeof(int),cudaMemcpyHostToDevice);
                    int found[100];
                    for(int y=0;y<100;y++)
                      found[y]=0;
                    int *d_found;
                    cudaMalloc(&d_found , 100*sizeof(int));
                    cudaMemcpy(d_found , found, 100*sizeof(int) , cudaMemcpyHostToDevice);

                    int *h_result[100];
                    int **d_result;
                    cudaMalloc(&d_result,100*sizeof(int*));
                   
                    find<<<100,7>>>(d_prefix_sum , d_tree , d_result , d_found  , tree.size() , n , d , d_keys);
                    cudaDeviceSynchronize();
                    
                    cudaMemcpy(h_result,d_result,100*sizeof(int*),cudaMemcpyDeviceToHost);
                    
                    cudaMemcpy(found , d_found , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                    
                    

                    for(int j=0;j<100;j++)
                    {
                          
                        if(found[j])
                        {
                            int * addr = h_result[j];
                            addr[find_keys[i+j][1]-1] = addr[find_keys[i+j][1]-1] + find_keys[i+j][2];
                            
                        }
                        else
                        {
                            //printf("-1\n");
                        }
                    }
                }
                int h_keys[100];
                for(int y=0;y<100;y++)
                {
                    h_keys[y]=-1;
                }
                idx=0;
                for(int i=(no_calls-1)*100;i<p;i++)
                {  
                    h_keys[idx]=find_keys[i][0];
                    idx++;
                }
                int *d_keys;
                cudaMalloc(&d_keys,100*sizeof(int));
                cudaMemcpy(d_keys,h_keys,100*sizeof(int),cudaMemcpyHostToDevice);

                int found[100];
                for(int y=0;y<100;y++)
                      found[y]=0;
                int *d_found;
                cudaMalloc(&d_found , 100*sizeof(int));  
                cudaMemcpy(d_found , found, 100*sizeof(int) , cudaMemcpyHostToDevice);
                
                int *h_result[100];
                int **d_result;
        
                cudaMalloc(&d_result,100*sizeof(int*));
                
                find<<<100,7>>>(d_prefix_sum , d_tree , d_result  , d_found ,  tree.size(), n , d , d_keys);
                 cudaDeviceSynchronize();
    
                cudaMemcpy(h_result,d_result , 100*sizeof(int*) ,cudaMemcpyDeviceToHost);
              
                cudaMemcpy(found , d_found , 100*(sizeof(int)) , cudaMemcpyDeviceToHost);
                
                
                idx = (no_calls-1)*100;
                for(int i=0;i<extra;i++)
                {
                    if(found[i])
                    {
                        int * addr = h_result[i];
                        addr[find_keys[i+idx][1] - 1] = addr[find_keys[i+idx][1] - 1] + find_keys[i+idx][2];
                        //printf("\n");
                        //fprintf( outputfilepointer, "\n");
                    }
                    else
                    {
                        //printf("-1\n");
                        //fprintf( outputfilepointer, "-1\n");
                    }
                }
            }
        }
        else
        {
            int key_;
            fscanf( inputfilepointer, "%d", &key_ );
            int kk[n];
            int *k;
            cudaMalloc(&k,n*sizeof(int));
            int h_count=0;
            int *d_count;
            cudaMalloc(&d_count,sizeof(int));
            cudaMemcpy(d_count,&h_count,sizeof(int),cudaMemcpyHostToDevice);
            path_trace<<<1,7>>>(d_prefix_sum , d_tree , k , d_count , tree.size() , n , d , key_);
            cudaMemcpy(&h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(kk,k,n*sizeof(int),cudaMemcpyDeviceToHost);
       
            for(int i=0;i<h_count;i++)
            {
                fprintf( outputfilepointer, "%d ",kk[i]);
                //printf("%d ",kk[i]);
            }
            //printf("\n");
            fprintf( outputfilepointer, "\n");
        }
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    fclose( outputfilepointer );
    fclose( inputfilepointer );
 
 
 
}


