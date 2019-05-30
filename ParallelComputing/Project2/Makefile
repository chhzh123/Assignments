ifdef LONG
INTT = -DLONG
endif

TEST = 10001x10001GraphExamples.txt
NUM_VERT = 20001

PCC = g++
PCFLAGS = -fopenmp -O3 -DOPENMP -Iinclude

PRG = sssp

all: $(PRG)

% : %.cpp
	$(PCC) $(PCFLAGS) $(INTT) $< -o $@

run:
	./$(PRG) $(TEST) $(NUM_VERT)

.PHONY: clean

clean:
	-rm $(PRG)