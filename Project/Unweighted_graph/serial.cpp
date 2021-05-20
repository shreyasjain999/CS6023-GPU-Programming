#include <bits/stdc++.h>
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

struct Graph{
    int nodes ;
    int edges;
    int *OA ;
    int *CA ;
    Graph(int n,int e){
        nodes = n ;
        edges = e ;
        OA = new int[nodes +1];
        CA = new int[2 * edges +1];
    }
};


void printTime(float ms) {
    int h = ms / (1000*3600);
    int m = (((int)ms) / (1000*60)) % 60;
    int s = (((int)ms) / 1000) % 60;
    int intMS = ms;
    intMS %= 1000;

    printf("Time Taken (Serial) = %dh %dm %ds %dms\n", h, m, s, intMS);
    printf("Time Taken in milliseconds : %d\n", (int)ms);
}

double *cal_BC(Graph *graph)
{
    int nodeCount = graph->nodes;
    
    double *bwC= new double[nodeCount]();
    vector<int> *predecessor = new vector<int>[nodeCount];

    double *dependency = new double[nodeCount];
    int *sigma = new int[nodeCount];
    int *dist = new int[nodeCount];

    // printf("Progress... %3d%%", 0);
    for (int s = 0; s < nodeCount; s++)
    {
        // printf("\rProgress... %5.2f%%", (s+1)*100.0/nodeCount);
        stack<int> st;
        
        memset(dist, -1, nodeCount * sizeof(int));
        memset(sigma, 0, nodeCount * sizeof(int));
        memset(dependency, 0, nodeCount * sizeof(double));

        dist[s] = 0;
        sigma[s] = 1;
        queue<int> q;
        q.push(s);
        while (!q.empty())
        {
            int v = q.front();
            q.pop();
            st.push(v);

            // For each neighbour w of v
            for (int i = graph->OA[v]; i < graph->OA[v + 1]; i++)
            {
                int w = graph->CA[i];
                // If w is visited for the first time
                if (dist[w] < 0)
                {
                    q.push(w);
                    dist[w] = dist[v] + 1;
                }
                // If shortest path to w from s goes through v
                if (dist[w] == dist[v] + 1)
                {
                    sigma[w] += sigma[v];
                    predecessor[w].push_back(v);
                }
            }
        }

        // st returns vertices in order of non-increasing distance from s
        while (!st.empty())
        {
            int w = st.top();
            st.pop();

            for (const int &v : predecessor[w])
            {
                if (sigma[w] != 0)
                    dependency[v] += (sigma[v] * 1.0 / sigma[w]) * (1 + dependency[w]);
            }
            if (w != s)
            {
                // Each shortest path is counted twice. So, each partial shortest path dependency is halved.
                bwC[w] += dependency[w] / 2;
            }
        }
        for(int i=0; i<nodeCount; ++i){
            predecessor[i].clear();
        }
        cout<<"Betweeness\n";
        for(int i=0; i<nodeCount; ++i){
            cout<<bwC[i]<<" ";

        }
        cout<<endl;
        cout<<endl;

    }

    delete[] predecessor, sigma, dependency, dist;
    cout << endl;
    return bwC;
}

int main(int argc, char *argv[])
{
    int m,n;
    int num1,num2;
    FILE *filePointer;
    char *filename = argv[1]; 
    filePointer = fopen( filename , "r") ; 
      
    //checking if file ptr is NULL
    if ( filePointer == NULL ) 
    {
        printf( "input.txt file failed to open." ) ; 
        return 0;
    }

    fscanf(filePointer, "%d", &n );     //scaning the number of vertices
    fscanf(filePointer, "%d", &m );     //scaning the number of edges

    Graph *graph = new Graph(n,m);

    vector <edgepairs> COO(2*m);
    int it=0;
    for(int i=0 ; i<m ; i++ )  //scanning the edges
    {
        fscanf(filePointer, "%d", &num1) ;
        fscanf(filePointer, "%d", &num2) ;
        COO[it].x = num1 ;
        COO[it].y = num2 ;
        it++;
        COO[it].x = num2 ;
        COO[it].y = num1 ;
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
        graph->CA[i] = COO[i].y;
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
   
    clock_t start, end;
    start = clock();

    double *bwCentrality = cal_BC(graph);

    end = clock();
    float time_taken = 1000.0 * (end - start) / (float)CLOCKS_PER_SEC;

    double maxBetweenness = -1;
    for (int i = 0; i < n; i++)
    {
        maxBetweenness = max(maxBetweenness, bwCentrality[i]);
        printf("Node %d => Betweeness Centrality %0.2lf\n", i, bwCentrality[i]);
    }

    cout << endl;
    printf("\nMaximum Betweenness Centrality ==> %0.2lf\n", maxBetweenness);
    printTime(time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < n; i++)
            cout << bwCentrality[i] << " ";
        cout << endl;
    }

    // Free all memory

    for(int i=0;i<2*m;i++)
    {
        cout<<COO[i].x << " "<<COO[i].y<<endl;
    }
    for(int i=0;i<n;i++)
    {
        cout<<graph->OA[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<2*m;i++)
    {
        cout<<graph->CA[i] << " ";
    }
    cout<<endl;
    cout<<time_taken<<endl;
    delete[] bwCentrality;
    delete graph;
    return 0;
}