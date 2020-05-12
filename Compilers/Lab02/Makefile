TARGET = lex

.PHONY: all clean test

all: $(TARGET)

$(TARGET): lex.yy.c
	gcc $< -o $@

lex.yy.c: $(TARGET).l
	lex $<

test: lex_test.cpp
	g++ lex_test.cpp -o lex_test

clean:
	-rm lex.yy.c $(TARGET) lex_test