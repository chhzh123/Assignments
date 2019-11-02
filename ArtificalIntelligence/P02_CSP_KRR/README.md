## Futoshiki
To run the GAC *python* program, please type

```bash
python futoshiki.py <input_file> <board_size>
```

For example, for the largest case, please enter

```bash
python futoshiki.py in5.txt 8
```

You should make sure the program and the input file are in the same folder.

* The input files are `in1.txt` - `in5.txt`.
    - Suppose the board size is n, then the first n lines are all the n*n numbers on the initial board. The empty grid uses 0 to denote. Different grids are separated by white space.
    - The next m lines denotes the constraints, using inequalities to represent. For example, `(1,2) < (2,2)` means the number in grid (1,2) should be smaller than the number in grid (2,2). The indices begin from 1.
* The results can be viewed in `futoshiki/caseX.png`.

## Blocks World
To run the blocks world *Prolog* program, you should first enter the interactive interface of Prolog, and type

```bash
?- ["blocksworld.pl","bf.pl","blocks1.pl"].
```

This example loads case 1. For other examples, you can modified the last file.

To query the planning result, you can type like this

```bash
?- end1(Goal), bestfirst(Goal->stop,Plan).
```

The queries are attached in the input files `blocksX.pl`, please check for that. The screenshots and the planning results can be found in `blocksworld/caseX.png` and `blocksworld/planX.txt`

You'd better reopen prolog dialog once you want to run a new case, making sure the previous facts have no impacts on the current case.