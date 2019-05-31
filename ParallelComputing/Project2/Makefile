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

TEST = 10001x10001GraphExamples.txt
NUM_VERT = 20001

ifdef MPI
PCC = mpic++
PCFLAGS = -fopenmp -O3 -DOPENMP -Iinclude -DMPIFLAG
else
PCC = g++
PCFLAGS = -fopenmp -O3 -DOPENMP -Iinclude
endif

PRG = sssp

all: $(PRG)

% : %.cpp
	$(PCC) $(PCFLAGS) $(INTT) $(INTE) $(DTIME) $(DDEBUG) $< -o $@

run:
	./$(PRG) $(TEST) $(NUM_VERT)

run_test:
	./$(PRG) test.txt 6

.PHONY: clean

clean:
	-rm $(PRG)