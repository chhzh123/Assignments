CC = g++
FLAGS = -fopenmp

ifdef DEBUG
DEBUG_FLAG = -DDEBUG
endif

ifdef NWORKER
NWORKER_FLAG = -DNWORKER=$(NWORKER)
endif

BENCHMARKS = gre saylr bus nos orsreg sherman orani bcsstk

all: matmul prod-cons

matmul: matmul.cpp
	$(CC) $(FLAGS) $(DEBUG_FLAG) $(NWORKER_FLAG) $< -o $@

prod-cons: prod-cons.cpp
	$(CC) $(FLAGS) $(DEBUG_FLAG) $(NWORKER_FLAG) $< -o $@

run:
	$(foreach dataset, $(BENCHMARKS), ./matmul benchmarks/$(dataset).mtx;)

run2:
	$(foreach num, 1 2 4 8, ./matmul benchmarks/bcsstk.mtx $(num);)

run3:
	./prod-cons

clean:
	-rm matmul prod-cons