import numpy as np
import time, sys

KERNEL_SIZE = 5
HALF_KERNEL = 2

cnt = 0
data = open("data.bin","rb")
samples = []
while True:
    chunk = data.read(4)
    if chunk == b'':
        break
    width = np.fromstring(chunk,dtype=np.uint32)[0]
    height = np.fromstring(data.read(4),dtype=np.uint32)[0]
    size = width * height
    samples.append(np.fromfile(data, count=size, dtype="uint8").reshape((height,width)))

def calculate(arr):
    height, width = arr.shape
    res = np.zeros(arr.shape)
    for i in range(height):
        for j in range(width):
            kernel = arr[max(0,i-HALF_KERNEL):min(height,i+HALF_KERNEL+1),
                         max(0,j-HALF_KERNEL):min(width,j+HALF_KERNEL+1)]
            size = kernel.shape[0] * kernel.shape[1]
            unique_elements, counts_elements = np.unique(kernel, return_counts=True)
            res[i][j] = -np.sum(counts_elements * np.log(counts_elements)) / size + np.log(size)
    return res

for sample in samples:
    height, width = sample.shape
    start = time.time()
    result = calculate(sample)
    end = time.time()
    print(sample.shape)
    print("sample:\n",sample[height-5:,width-5:])
    print("result:\n",result[height-5:,width-5:])
    print("numpyCallback: {:.2f} ms\n".format((end - start) * 1000))