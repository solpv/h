#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

// Structure to represent an undirected graph
struct Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list representation
};

// Add edge to the graph
void addEdge(Graph &graph, int u, int v) {
    graph.adj[u].push_back(v);
    graph.adj[v].push_back(u);
}

// Parallel Breadth First Search algorithm
void parallelBFS(Graph &graph, int source) {
    int V = graph.V;
    vector<bool> visited(V, false);
    queue<int> q;
    visited[source] = true;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        cout << u << " "; // Process the current node
#pragma omp parallel for
        for (int i = 0; i < graph.adj[u].size(); i++) {
            int v = graph.adj[u][i];
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
}

// Parallel Depth First Search algorithm
void parallelDFS(Graph &graph, int source) {
    int V = graph.V;
    vector<bool> visited(V, false);
    stack<int> stk;
    stk.push(source);

    while (!stk.empty()) {
        int u = stk.top();
        stk.pop();
        if (!visited[u]) {
            cout << u << " "; // Process the current node
            visited[u] = true;
#pragma omp parallel for
            for (int i = 0; i < graph.adj[u].size(); i++) {
                int v = graph.adj[u][i];
                if (!visited[v]) {
                    stk.push(v);
                }
            }
        }
    }
}

int main() {
    Graph graph;
    int V, E; // Number of vertices and edges
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;

    graph.V = V;
    graph.adj.resize(V);

    cout << "Enter the edges (format: source destination):" << endl;
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        addEdge(graph, u, v);
    }

    int choice;
    cout << "Choose Search Algorithm: " << endl;
    cout << "1. Breadth First Search (BFS)" << endl;
    cout << "2. Depth First Search (DFS)" << endl;
    cout << "3. Both BFS and DFS" << endl;
    cout << "Enter your choice (1, 2, or 3): ";
    cin >> choice;

    if (choice == 1) {
        cout << "Breadth First Search Traversal: ";
        parallelBFS(graph, 0); // Start BFS from node 0
        cout << endl;
    } else if (choice == 2) {
        cout << "Depth First Search Traversal: ";
        parallelDFS(graph, 0); // Start DFS from node 0
        cout << endl;
    } else if (choice == 3) {
        cout << "Breadth First Search Traversal: ";
        parallelBFS(graph, 0); // Start BFS from node 0
        cout << endl;
        cout << "Depth First Search Traversal: ";
        parallelDFS(graph, 0); // Start DFS from node 0
        cout << endl;
    } else {
        cout << "Invalid choice!" << endl;
    }

    return 0;
}

