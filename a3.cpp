#include <iostream>
#include <omp.h>
#include <limits>

using namespace std;

// Function to find minimum element using parallel reduction
int parallelMin(int arr[], int n) {
    int min_val = numeric_limits<int>::max();
#pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

// Function to find maximum element using parallel reduction
int parallelMax(int arr[], int n) {
    int max_val = numeric_limits<int>::min();
#pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// Function to find sum using parallel reduction
int parallelSum(int arr[], int n) {
    int sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Function to find average using parallel reduction
double parallelAverage(int arr[], int n) {
    int sum = parallelSum(arr, n);
    return static_cast<double>(sum) / n;
}

int main() {
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    int arr[n];
    cout << "Enter " << n << " elements of the array: ";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    // Find and print minimum element
    int min_val = parallelMin(arr, n);
    cout << "Minimum value: " << min_val << endl;

    // Find and print maximum element
    int max_val = parallelMax(arr, n);
    cout << "Maximum value: " << max_val << endl;

    // Find and print sum
    int sum = parallelSum(arr, n);
    cout << "Sum: " << sum << endl;

    // Find and print average
    double average = parallelAverage(arr, n);
    cout << "Average: " << average << endl;

    return 0;
}


