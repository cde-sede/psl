from enum import Enum, auto
import uuid
from typing import Optional, Any, Iterator, TextIO
from io import StringIO
import collections
import re


class LangExceptions(Exception):
	pass

class UnknownToken(LangExceptions): pass
class InvalidSyntax(LangExceptions): pass
class SymbolRedefined(LangExceptions): pass
class FileError(LangExceptions): pass

class TypesType(Enum):
	pass
class PreprocTypes(TypesType):
	MACRO		 = auto()
	INCLUDE		 = auto()

class FlowControl(TypesType):
	OP_IF		 = auto()
	OP_ELSE		 = auto()
	OP_WHILE	 = auto()
	OP_DO		 = auto()
	OP_END		 = auto()
	OP_LABEL	 = auto()

class Operands(TypesType):
	OP_PLUS		 = auto()
	OP_MINUS	 = auto()
	OP_MUL		 = auto()
	OP_DIV		 = auto()
	OP_MOD		 = auto()
	OP_DIVMOD	 = auto()

	OP_BLSH		 = auto()
	OP_BRSH		 = auto()
	OP_BAND		 = auto()
	OP_BOR		 = auto()
	OP_BXOR		 = auto()

	OP_EQ		 = auto()
	OP_NE		 = auto()
	OP_GT		 = auto()
	OP_GE		 = auto()
	OP_LT		 = auto()
	OP_LE		 = auto()

class Intrinsics(TypesType):
	OP_DROP		 = auto()
	OP_DUP		 = auto()
	OP_DUP2		 = auto()
	OP_SWAP		 = auto()
	OP_OVER		 = auto()
	OP_MEM		 = auto()

class OpTypes(TypesType):
	OP_PUSH		 = auto()
	OP_STRING	 = auto()

	OP_WORD		 = auto()

	OP_STORE	 = auto()
	OP_LOAD		 = auto()
	OP_STORE16	 = auto()
	OP_LOAD16	 = auto()
	OP_STORE32	 = auto()
	OP_LOAD32	 = auto()
	OP_STORE64	 = auto()
	OP_LOAD64	 = auto()

	OP_DUMP		 = auto()
	OP_CDUMP	 = auto()
	OP_UDUMP	 = auto()
	OP_HEXDUMP	 = auto()

	OP_SYSCALL	 = auto()
	OP_SYSCALL1	 = auto()
	OP_SYSCALL2	 = auto()
	OP_SYSCALL3	 = auto()
	OP_SYSCALL4	 = auto()
	OP_SYSCALL5	 = auto()
	OP_SYSCALL6	 = auto()

	OP_RSYSCALL1 = auto()
	OP_RSYSCALL2 = auto()
	OP_RSYSCALL3 = auto()
	OP_RSYSCALL4 = auto()
	OP_RSYSCALL5 = auto()
	OP_RSYSCALL6 = auto()

	OP_EXIT		 = auto()
	OP_COUNT	 = auto()


class TokenTypes(Enum):
	NUMBER		 = auto()
	STRING		 = auto()
	CHAR		 = auto()
	OP			 = auto()
	WORD		 = auto()
	NEW_LINE	 = auto()

	TOKEN_COUNT	 = auto()

class TokenInfo(collections.namedtuple("TokenInfo", "type string start end line")):
	type: TokenTypes
	string: str
	start: tuple[int, int]
	end: tuple[int, int]
	line: str

	def __repr__(self) -> str:
		return f"TokenInfo(type={self.type}, string={self.string!r}, start={self.start!r}, end={self.end!r}, line={self.line!r})"

	def error(self) -> str:
		return f"{self.line}{'': <{self.start[1]}}{'':^<{self.end[1] - self.start[1]}}"

class Token:
	__slots__ = ("type", "value", "info", "id", "position")

	type: TypesType
	value: Any
	info: Optional[TokenInfo]
	id: str
	position: int

	def __init__(self, type: TypesType, value: Any=None, info=None):
		self.value = value
		self.type = type
		self.info = info
		self.id = str(uuid.uuid4())[:8]
		self.position = -1

	def __repr__(self) -> str:
		return f"{self.type} {f" {self.value}" if self.value else ""}"

	def label(self) -> str:
		return f"{self.type.name}_{self.id}"

	def copy(self) -> 'Token':
		return Token(
			value=self.value,
			type=self.type,
			info=self.info,
		)

NUMBER_REG	= re.compile(r"^(\s*)(-?\d+)")
STRING_REG	= re.compile(r"^(\s*)\"(.*)\"")
OP_REG		= re.compile(r"^(\s*)((?:[^\w\s]|\d)+)")
CHAR_REG	= re.compile(r"^(\s*)'(\\?.)'")
WORD_REG	= re.compile(r"^(\s*)(\w+)")
ANY_REG		= re.compile(r"^(\s*)(.+)")

def replace_tabs(s: str) -> Iterator[str]:
	j = 0
	for i,c in enumerate(s):
		if c == '\t':
			for k in range(4 - j % 4):
				yield ' '
			j = 0
		else:
			yield c
			j = (j + 1) % 4

def _tokenize(f, *, debug=False) -> Iterator[TokenInfo]:
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

class Tokenize:
	def __init__(self, buffer: TextIO, *, debug=False, close=False):
		self.buffer: TextIO = buffer
		self.debug: bool = debug
		self.close: bool = close

		self._extend: list[Iterator] = []
		

	def __iter__(self):
		self._tokenize = _tokenize(self.buffer, debug=self.debug)
		return self

	def __next__(self) -> TokenInfo:
		if self._extend:
			try:
				return next(self._extend[0])
			except StopIteration:
				self._extend.pop(0)
		try:
			t = next(self._tokenize)
		except StopIteration:
			if self.close:
				self.buffer.close()
			raise StopIteration
		return t

	def extend(self, tokens):
		self._extend.append(tokens)
