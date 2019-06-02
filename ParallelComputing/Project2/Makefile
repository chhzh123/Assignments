ifdef LONG
INTT = -DLONG
endif

ifdef EDGELONG
INTE = -DEDGELONG
endif

ifdef TIME
DTIME = -DTIME
endif

ifdef DEBUG
DDEBUG = -DDEBUG -g
endif

ifdef DG
GFLAG = -g
endif

TEST = 10001x10001GraphExamples.txt
NUM_VERT = 20001

ifdef MPI
PCC = mpic++
PCFLAGS = -fopenmp -O3 -DOPENMP -Iinclude -DMPIFLAG -std=c++11
else
PCC = g++
PCFLAGS = -fopenmp -O3 -DOPENMP -Iinclude -std=c++11
endif

MPIRUN = mpirun
NUM_PROC = 3

PRG = sssp

all: $(PRG)

% : %.cpp
	$(PCC) $(PCFLAGS) $(INTT) $(INTE) $(DTIME) $(DDEBUG) $(GFLAG) $< -o $@

run:
	./$(PRG) $(TEST) $(NUM_VERT)

run_mpi:
	$(MPIRUN) -n $(NUM_PROC) ./$(PRG) $(TEST) $(NUM_VERT)

run_test:
	./$(PRG) test.txt 6

run_mpi_test:
	$(MPIRUN) -n $(NUM_PROC) ./$(PRG) test.txt 6

.PHONY: clean

clean:
	-rm $(PRG)