CC=gcc
CFLAGS=-I. -g
DEPS=
ODIR=obj

all: classify

$(ODIR)/%.o: %.cpp $(DEPS)
	@mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

classify: $(ODIR)/classify.o
	$(CC) -std=c++11 -o $@ $^ -lstdc++

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
