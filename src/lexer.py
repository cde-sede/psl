from enum import Enum
import uuid
from typing import Optional, Any
from io import StringIO
import collections
import re


class LangExceptions(Exception):
	pass

class UnknownToken(LangExceptions):
	pass
class InvalidSyntax(LangExceptions):
	pass
class SymbolRedefined(LangExceptions):
	pass


def iota(reset=False, *, v=[-1]):
	if reset:
		v[0] = -1
	v[0] += 1
	return v[0]


class OpTypes(Enum):
	OP_PUSH		 = iota(True)
	OP_STRING	 = iota()
	OP_WORD		 = iota()
	OP_DROP		 = iota()
	OP_DUP		 = iota()
	OP_DUP2		 = iota()
	OP_SWAP		 = iota()
	OP_OVER		 = iota()

	OP_PLUS		 = iota()
	OP_MINUS	 = iota()
	OP_MUL		 = iota()
	OP_DIV		 = iota()
	OP_MOD		 = iota()
	OP_DIVMOD	 = iota()

	OP_BLSH		 = iota()
	OP_BRSH		 = iota()
	OP_BAND		 = iota()
	OP_BOR		 = iota()
	OP_BXOR		 = iota()

	OP_EQ		 = iota()
	OP_NE		 = iota()
	OP_GT		 = iota()
	OP_GE		 = iota()
	OP_LT		 = iota()
	OP_LE		 = iota()

	OP_IF		 = iota()
	OP_ELSE		 = iota()
	OP_WHILE	 = iota()
	OP_DO		 = iota()
	OP_MACRO	 = iota()
	OP_END		 = iota()
	OP_LABEL	 = iota()

	OP_MEM		 = iota()
	OP_STORE	 = iota()
	OP_LOAD		 = iota()

	OP_DUMP		 = iota()
	OP_CDUMP	 = iota()
	OP_UDUMP	 = iota()
	OP_HEXDUMP	 = iota()
	OP_PRINTLINE = iota()
	OP_SYSCALL	 = iota()
	OP_SYSCALL1	 = iota()
	OP_SYSCALL2	 = iota()
	OP_SYSCALL3	 = iota()
	OP_SYSCALL4	 = iota()
	OP_SYSCALL5	 = iota()
	OP_SYSCALL6	 = iota()

	OP_RSYSCALL1 = iota()
	OP_RSYSCALL2 = iota()
	OP_RSYSCALL3 = iota()
	OP_RSYSCALL4 = iota()
	OP_RSYSCALL5 = iota()
	OP_RSYSCALL6 = iota()

	OP_EXIT		 = iota()
	OP_COUNT	 = iota()


class TokenTypes(Enum):
	NUMBER		 = iota(True)
	STRING		 = iota()
	CHAR		 = iota()
	OP			 = iota()
	WORD		 = iota()
	NEW_LINE	 = iota()

	TOKEN_COUNT	 = iota()

class TokenInfo(collections.namedtuple("TokenInfo", "type string start end line")):
	type: TokenTypes
	string: str
	start: tuple[int, int]
	end: tuple[int, int]
	line: str

	def __repr__(self):
		return f"TokenInfo(type={self.type}, string={self.string!r}, start={self.start!r}, end={self.end!r}, line={self.line!r})"

	def error(self):
		return f"{self.line}{'': <{self.start[1]}}{'':^<{self.end[1] - self.start[1]}}"

class Token:
	__slots__ = ("type", "value", "info", "id", "position")

	type: OpTypes
	value: Any
	info: Optional[TokenInfo]
	id: str
	position: int

	def __init__(self, type: OpTypes, value: Any=None, info=None):
		self.value = value
		self.type = type
		self.info = info
		self.id = str(uuid.uuid4())[:8]
		self.position = -1

	def __repr__(self):
		return f"{self.type}" #{f" {self.value}" if self.value else ""}"

	def label(self):
		return f"{self.type.name}_{self.id}"

	def copy(self):
		return Token(
			value=self.value,
			type=self.type,
			info=self.info,
		)

NUMBER_REG	= re.compile(r"^(\s*)(\d+)")
STRING_REG	= re.compile(r"^(\s*)\"(.*)\"")
OP_REG		= re.compile(r"^(\s*)([^\w\s]+)")
CHAR_REG	= re.compile(r"^(\s*)'(\\?.)'")
WORD_REG	= re.compile(r"^(\s*)(\w+)")
ANY_REG		= re.compile(r"^(\s*)(.+)")

def replace_tabs(s):
	j = 0
	for i,c in enumerate(s):
		if c == '\t':
			for k in range(4 - j % 4):
				yield ' '
			j = 0
		else:
			yield c
			j = (j + 1) % 4

def tokenize(f, *, debug=False):
	toks = []
	for line_number, line in enumerate(f):
		if not line.strip(): continue
		line = ''.join(replace_tabs(line))
		index = 0
		while index < len(line):
			if r := re.match(NUMBER_REG, line[index:]):
				yield (t := TokenInfo(
					TokenTypes.NUMBER,
					r.groups()[1],
					(line_number, index + len(r.groups()[0])),
					(line_number, index + r.span()[1]),
					line,
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(STRING_REG, line[index:]):
				yield (t := TokenInfo(
					TokenTypes.STRING,
					r.groups()[1],
					(line_number, index + len(r.groups()[0])),
					(line_number, index + r.span()[1]),
					line,
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(CHAR_REG, line[index:]):
				yield (t := TokenInfo(
					TokenTypes.CHAR,
					r.groups()[1],
					(line_number, index + len(r.groups()[0])),
					(line_number, index + r.span()[1]),
					line,
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(OP_REG, line[index:]):
				yield (t := TokenInfo(
					TokenTypes.OP,
					r.groups()[1],
					(line_number, index + len(r.groups()[0])),
					(line_number, index + r.span()[1]),
					line,
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(WORD_REG, line[index:]):
				yield (t := TokenInfo(
					TokenTypes.WORD,
					r.groups()[1],
					(line_number, index + len(r.groups()[0])),
					(line_number, index + r.span()[1]),
					line,
				))
				if debug: print(t)
				index += r.span()[1]
			else:
				if line[index:].strip():
					r = re.match(ANY_REG, line[index:])
					if not r:
						raise ValueError(line, index)
					raise InvalidSyntax(TokenInfo(
						TokenTypes.WORD,
						r.groups()[1],
						(line_number, index + len(r.groups()[0])),
						(line_number, index + r.span()[1]),
						line,
					))
				else:
					yield (t := TokenInfo(
						TokenTypes.NEW_LINE,
						'\n', (line_number, index), (line_number, index + 1), line
					))
					if debug: print(t)
				break

if __name__ == '__main__':
	import sys
	with open(sys.argv[1], 'r') as f:
		toks = tokenize(f)

		print(toks[13])
