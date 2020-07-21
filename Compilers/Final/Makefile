ifdef NO_STDIN
FLAGS = -DNO_STDIN
endif

all: regex

regex: main.cpp
	g++ $< -g $(FLAGS) -o $@

.PHONY: clean
clean:
	rm regex