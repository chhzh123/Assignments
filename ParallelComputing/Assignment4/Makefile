CC = nvcc
PRG = error-test error-test-2 jacobi

all: $(PRG)

% : %.cu
	$(CC) $< -o $@

run1:
	./error-test

run2:
	./jacobi

run3:
	./error-test-2

.PHONY: clean
clean:
	-rm -rf $(PRG)