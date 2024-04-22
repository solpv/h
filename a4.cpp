#include <iostream>
#include <vector>

#define TILE_SIZE 16

using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAddition(int *a, int *b, int *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(int *a, int *b, int *c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

int main() {
    int vectorSize, rowsA, colsA, colsB;

    // User input for vector size and matrix dimensions
    cout << "Enter the size of the vectors: ";
    cin >> vectorSize;
    cout << "Enter the number of rows for matrix A: ";
    cin >> rowsA;
    cout << "Enter the number of columns for matrix A: ";
    cin >> colsA;
    cout << "Enter the number of columns for matrix B: ";
    cin >> colsB;

    // Allocate memory for vectors and matrices on host
    vector<int> h_a(vectorSize);
    vector<int> h_b(vectorSize);
    vector<int> h_c_vector(vectorSize);
    vector<int> h_a_matrix(rowsA * colsA);
    vector<int> h_b_matrix(colsA * colsB);
    vector<int> h_c_matrix(rowsA * colsB);

    // Initialize vectors with random values
    for (int i = 0; i < vectorSize; ++i) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Initialize matrices with random values
    for (int i = 0; i < rowsA * colsA; ++i) {
        h_a_matrix[i] = rand() % 10;
    }
    for (int i = 0; i < colsA * colsB; ++i) {
        h_b_matrix[i] = rand() % 10;
    }

    // Allocate memory for vectors and matrices on device
    int *d_a, *d_b, *d_c_vector, *d_a_matrix, *d_b_matrix, *d_c_matrix;
    cudaMalloc(&d_a, vectorSize * sizeof(int));
    cudaMalloc(&d_b, vectorSize * sizeof(int));
    cudaMalloc(&d_c_vector, vectorSize * sizeof(int));
    cudaMalloc(&d_a_matrix, rowsA * colsA * sizeof(int));
    cudaMalloc(&d_b_matrix, colsA * colsB * sizeof(int));
    cudaMalloc(&d_c_matrix, rowsA * colsB * sizeof(int));

    // Copy vectors and matrices from host to device
    cudaMemcpy(d_a, h_a.data(), vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_matrix, h_a_matrix.data(), rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_matrix, h_b_matrix.data(), colsA * colsB * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for vector addition
    int blockSize_vector = 256;
    int gridSize_vector = (vectorSize + blockSize_vector - 1) / blockSize_vector;

    // Launch vector addition kernel
    vectorAddition<<<gridSize_vector, blockSize_vector>>>(d_a, d_b, d_c_vector, vectorSize);

    // Copy result vector from device to host
    cudaMemcpy(h_c_vector.data(), d_c_vector, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Define grid and block dimensions for matrix multiplication
    dim3 blockSize_matrix(TILE_SIZE, TILE_SIZE);
    dim3 gridSize_matrix((colsB + TILE_SIZE - 1) / TILE_SIZE, (rowsA + TILE_SIZE - 1) / TILE_SIZE);

    // Launch matrix multiplication kernel
    matrixMultiplication<<<gridSize_matrix, blockSize_matrix>>>(d_a_matrix, d_b_matrix, d_c_matrix, rowsA, colsA, colsB);

    // Copy result matrix from device to host
    cudaMemcpy(h_c_matrix.data(), d_c_matrix, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_vector);
    cudaFree(d_a_matrix);
    cudaFree(d_b_matrix);
    cudaFree(d_c_matrix);

    // Display the results
    cout << "Result of vector addition:\n";
    for (int i = 0; i < vectorSize; ++i) {
        cout << h_c_vector[i] << " ";
    }
    cout << endl;

    cout << "Result of matrix multiplication:\n";
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            cout << h_c_matrix[i * colsB + j] << " ";
        }
        cout << endl;
    }

    return 0;
}


