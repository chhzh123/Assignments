CC = g++
FLAGS = -fopenmp

ifdef DEBUG
DEBUG_FLAG = -DDEBUG
endif

ifdef NWORKER
NWORKER_FLAG = -DNWORKER=$(NWORKER)
endif

BENCHMARKS = gre saylr bus nos orsreg sherman orani bcsstk

all: matmul

matmul: matmul.cpp
	$(CC) $(FLAGS) $(DEBUG_FLAG) $(NWORKER_FLAG) $< -o $@

run:
	$(foreach dataset, $(BENCHMARKS), ./matmul benchmarks/$(dataset).mtx;)

run2:
	$(foreach num, 1 2 4 8, ./matmul benchmarks/bcsstk.mtx $(num);)

clean:
	-rm matmul