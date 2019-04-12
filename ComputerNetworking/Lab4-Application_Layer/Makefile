CC = gcc
FLAGS = -pthread
ALL = client ftp ftpDownload

all: $(ALL)

% : %.c
	$(CC) $(FLAGS) $< -o $@

.PHONY : clean

clean :
	-rm -f *.o $(ALL)