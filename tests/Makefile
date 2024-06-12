SRC = $(wildcard *.pyslang)
EXE = $(SRC:%.pyslang=%)

command = python3.12 test.py 
args = $(EXE:%=-s%)

all: test

test: $(EXE)
	${command} test -s ${EXE}

save: $(EXE)

%: %.pyslang
	${command} save -s $< -o $@ -a'1' -a'2' -a'3' -a'4'

fclean:
	rm $(EXE)
