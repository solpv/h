#include <iostream>
#include <omp.h>

using namespace std;

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0;
    int j = 0;
    int k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}


void parallelBubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
#pragma omp parallel for shared(swapped)
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }
}


void parallelMergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(arr, l, m);
#pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}


void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int choice;
    cout << "Choose Sorting Algorithm:" << endl;
    cout << "1. Parallel Bubble Sort" << endl;
    cout << "2. Parallel Merge Sort" << endl;
    cout << "3. Both Parallel Bubble Sort and Parallel Merge Sort" << endl;
    cout << "Enter your choice (1, 2, or 3): ";
    cin >> choice;

    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr) / sizeof(arr[0]);

    int tempArr[n]; 

    switch (choice) {
        case 1: {
            cout << "Original array: ";
            printArray(arr, n);

           
            parallelBubbleSort(arr, n);

            cout << "Sorted array (Parallel Bubble Sort): ";
            printArray(arr, n);
            break;
        }
        case 2: {
            cout << "Original array: ";
            printArray(arr, n);

           
            parallelMergeSort(arr, 0, n - 1);

            cout << "Sorted array (Parallel Merge Sort): ";
            printArray(arr, n);
            break;
        }
        case 3: {
            cout << "Original array: ";
            printArray(arr, n);

            
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    #pragma omp task
                    parallelBubbleSort(arr, n);
                }
                #pragma omp section
                {
                    #pragma omp task
                    parallelMergeSort(arr, 0, n - 1);
                }
            }

            cout << "Sorted array (Parallel Bubble Sort): ";
            printArray(arr, n);
            cout << "Sorted array (Parallel Merge Sort): ";
            printArray(arr, n);
            break;
        }
        default:
            cout << "Invalid choice!" << endl;
            break;
    }

    return 0;
}


