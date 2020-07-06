import numpy as np
import time, sys, os

KERNEL_SIZE = 5
KERNEL_RADIUS = 2

def read_data(filename="../data.bin"):
    """
    Read data from binary file,
    and return a list of 2D numpy arrays
    """
    data = open(filename,"rb")
    samples = []
    while True:
        # the first two int (4B) are matrix size
        # the elements in the matrix are all stored in 1B
        chunk = data.read(4)
        if chunk == b'':
            break
        width = np.fromstring(chunk,dtype=np.uint32)[0]
        height = np.fromstring(data.read(4),dtype=np.uint32)[0]
        size = width * height
        samples.append(np.fromfile(data, count=size, dtype="uint8").reshape((height,width)))
    return samples

def calculate(mat):
    """
    Calculate the central entropy of the input matrix

    Parameters:
    -----------
    mat : A 2D numpy array

    Return:
    -----------
    res : The central entropy matrix of mat
    """
    height, width = mat.shape
    res = np.zeros(mat.shape)
    for i in range(height):
        for j in range(width):
            # extract window
            kernel = mat[max(0,i-KERNEL_RADIUS):min(height,i+KERNEL_RADIUS+1),
                         max(0,j-KERNEL_RADIUS):min(width,j+KERNEL_RADIUS+1)]
            size = kernel.shape[0] * kernel.shape[1]
            # count histogram
            unique_elements, counts_elements = np.unique(kernel, return_counts=True)
            res[i][j] = -np.sum(counts_elements * np.log(counts_elements)) / size + np.log(size)
    return res

def print_mat(mat):
    """
    Output results
    """
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print("{:>8.5f}".format(mat[i][j]), end=" ")
        print()

if os.path.isfile("results.csv"):
    os.remove("results.csv")
outfile = open("results.csv","a")
samples = read_data()
for sample in samples:
    height, width = sample.shape
    start = time.time()
    result = calculate(sample)
    end = time.time()
    print(sample.shape)
    print("sample:")
    print_mat(sample[height-5:,width-5:])
    print("result:")
    print_mat(result[height-5:,width-5:])
    print("numpyCallback: {:.2f} ms\n".format((end - start) * 1000))
    np.savetxt(outfile,result.reshape(-1),"%.5f",delimiter=",")