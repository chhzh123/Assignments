# Scaffold for hw2

## How about the layout of the test samples?

In each test sample, there exist three parts:
- the first part contains the meta data of the sample;
- the second part contains the search points; and
- the third part contains the reference points.

To be more specific, we give a simple example containing only two search points
and two reference points in 3-dimensional space:

```python
k = 3
m = 1
n = 2

searchPoints = [
  1.0, 1.0, 0.1,  # m0
  0.4, 0.0, 0.0,  # m1
]

referencePoints = [
  0.5, 0.0, 0.0,  # n0
  1.0, 1.0, 0.0,  # n1
]
```

We conclude that:
- a test sample is indeed two arrays of float numbers with some meta data;
- the first three integers, i.e., `k`, `m`, and `n`, make up the meta data part,
  with `k` being the dimension size, `m` the number of search points, and
  `n` the number of reference points;
- then there are `m` points in the searchPoints array, with every `k`
  consecutive numbers representing a search point;
- and it is similar for the referencePoints array;
- besides, all of search points and reference points reside in `[0, 1]^k`.

As required in the slides for hw2, we have to find out the nearest point of each
search point and return their indices. We assume the reference points are
indexed starting from 0. And for search points, we should preserve their order
when writing out the correspondent results.

So by calculating the Euclidean distance between points, we should return
`[1, 0]` for the example above.

## Requirements

Further requirements about how to submit your solutions to hw2 will be announced
later.