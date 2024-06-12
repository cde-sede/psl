SRC = $(wildcard *.pyslang)
EXE = $(SRC:%.pyslang=%)

all: $(EXE)

%: %.pyslang
	@python3.12 -m src test -s $< -I./src/std

fclean:
	rm $(EXE)
