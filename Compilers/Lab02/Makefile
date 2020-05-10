TARGET = test

all: $(TARGET)

$(TARGET): lex.yy.c
	gcc $< -o $@

lex.yy.c: $(TARGET).l
	lex $<

.PHONY: clean
clean:
	-rm lex.yy.c $(TARGET)