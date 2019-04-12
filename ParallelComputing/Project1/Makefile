CC = g++
FLAGS = -std=c++11 -pthread

ifdef DEBUG
DEBUGFLAG = -g
endif

all: queue

% : %.cpp
	$(CC) $(FLAGS) $(DEBUGFLAG) $< -o $@

.PHONY: clean
clean:
	-rm -rf queue

run:
	./queue