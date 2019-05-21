CC = gcc
FLAGS =
ALL = server client

all: $(ALL)

% : %.c
	$(CC) $(FLAGS) $< -o $@

.PHONY : clean

clean :
	-rm -f *.o $(ALL)