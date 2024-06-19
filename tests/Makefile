SRC = $(wildcard *.pyslang)
EXE = $(SRC:%.pyslang=%)
DAT = $(SRC:%.pyslang=%.out)

command = python3.12 test.py
args = $(SRC:%.pyslang=-s='%.out')

all: test

test: $(DAT)
	${command} test ${args}

show: $(DAT)
	${command} show ${args}

save: $(DAT)

%.out: %.pyslang
	${command} save -s $< -o $@ -a'1' -a'2' -a'3' -a'4'

re: fclean all

fclean:
	rm -f $(DAT)
	rm -f $(EXE)

clean:
	rm -f $(EXE)
