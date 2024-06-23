SRC = $(wildcard *.pyslang)
EXE = $(SRC:%.pyslang=%)
DAT = $(SRC:%.pyslang=%.out)

command = python3.12 test.py
testargs = $(SRC:%.pyslang=-s'%.out')
saveargs = $(SRC:%.pyslang=-s'%.pyslang')

all: test

test:
	${command} test ${testargs}

show:
	${command} show ${testargs}

save:
	${command} save ${saveargs} -a'1' -a'2' -a'3' -a'4'

re: fclean all

fclean:
	rm -f $(DAT)
	rm -f $(EXE)

clean:
	rm -f $(EXE)
