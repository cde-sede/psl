from enum import Enum, auto
import uuid
from typing import Optional, Any, Iterator, TextIO
from io import StringIO
import collections
import re
from dataclasses import dataclass
from copy import deepcopy

from .errors import (
	InvalidSyntax,
)

from .classes import TypesType, TokenInfo


class PreprocTypes(TypesType):
	MACRO		 = auto()
	CALL		 = auto()
	PROC		 = auto()
	IN			 = auto()
	OUT			 = auto()
	MEMORY		 = auto()
	INCLUDE		 = auto()
	CAST		 = auto()

class FlowControl(TypesType):
	OP_IF		 = auto()
	OP_ELIF		 = auto()
	OP_ELSE		 = auto()
	OP_WHILE	 = auto()
	OP_DO		 = auto()
	OP_END		 = auto()
	OP_LABEL	 = auto()
	OP_LET		 = auto()
	OP_WITH		 = auto()

class Operands(TypesType):
	OP_PLUS		 = auto()
	OP_MINUS	 = auto()
	OP_MUL		 = auto()
	OP_DIV		 = auto()
	OP_MOD		 = auto()
	OP_DIVMOD	 = auto()

	OP_INCREMENT = auto()
	OP_DECREMENT = auto()

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
	OP_SWAP2	 = auto()
	OP_OVER		 = auto()
	OP_ROT		 = auto()
	OP_RROT		 = auto()
	OP_ARGC		 = auto()
	OP_ARGV		 = auto()

class OpTypes(TypesType):
	OP_PUSH		 = auto()
	OP_STRING	 = auto()
	OP_CHAR		 = auto()
	OP_BOOL		 = auto()

	OP_PUSHMEMORY= auto()

	OP_WORD		 = auto()

	OP_STORE	 = auto()
	OP_STORE16	 = auto()
	OP_STORE32	 = auto()
	OP_STORE64	 = auto()

	OP_LOAD		 = auto()
	OP_LOAD16	 = auto()
	OP_LOAD32	 = auto()
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


class TokenTypes(Enum):
	NUMBER		 = auto()
	STRING		 = auto()
	CHAR		 = auto()
	OP			 = auto()
	WORD		 = auto()
	NEW_LINE	 = auto()
	CAST		 = auto()

	TOKEN_COUNT	 = auto()


NUMBER_REG	 = re.compile(r"^(\s*)(-?\d+)(\s+|$)")
STRING_REG	 = re.compile(r"^(\s*)\"(.*?)\"(\s+|$)")
OP_REG		 = re.compile(r"^(\s*)((?:[^\w\s]|\d)+)(\s+|$)")
CHAR_REG	 = re.compile(r"^(\s*)'(\\?.)'(\s+|$)")
CAST_REG	 = re.compile(r"^(\s*):(\w+\**)(\s+|$)")
WORD_REG	 = re.compile(r"^(\s*)([^\s]+)(\s+|$)")
ANY_REG		 = re.compile(r"^(\s*)(.+)(\s+|$)")

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
					type=TokenTypes.NUMBER,
					string=r.groups()[1],
					start=(line_number, index + len(r.groups()[0])),
					end=(line_number, index + r.span()[1]),
					line=line,
					file=f.name
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(STRING_REG, line[index:]):
				yield (t := TokenInfo(
					type=TokenTypes.STRING,
					string=r.groups()[1],
					start=(line_number, index + len(r.groups()[0])),
					end=(line_number, index + r.span()[1]),
					line=line,
					file=f.name
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(CHAR_REG, line[index:]):
				yield (t := TokenInfo(
					type=TokenTypes.CHAR,
					string=r.groups()[1],
					start=(line_number, index + len(r.groups()[0])),
					end=(line_number, index + r.span()[1]),
					line=line,
					file=f.name
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(CAST_REG, line[index:]):
				yield (t := TokenInfo(
					type=TokenTypes.CAST,
					string=r.groups()[1],
					start=(line_number, index + len(r.groups()[0])),
					end=(line_number, index + r.span()[1]),
					line=line,
					file=f.name
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(OP_REG, line[index:]):
				yield (t := TokenInfo(
					type=TokenTypes.OP,
					string=r.groups()[1],
					start=(line_number, index + len(r.groups()[0])),
					end=(line_number, index + r.span()[1]),
					line=line,
					file=f.name
				))
				if debug: print(t)
				index += r.span()[1]
			elif r := re.match(WORD_REG, line[index:]):
				yield (t := TokenInfo(
					type=TokenTypes.WORD,
					string=r.groups()[1],
					start=(line_number, index + len(r.groups()[0])),
					end=(line_number, index + r.span()[1]),
					line=line,
					file=f.name
				))
				if debug: print(t)
				index += r.span()[1]
			else:
				if line[index:].strip():
					r = re.match(ANY_REG, line[index:])
					if not r:
						raise ValueError(line, index)
					raise InvalidSyntax(TokenInfo(
						type=TokenTypes.WORD,
						string=r.groups()[1],
						start=(line_number, index + len(r.groups()[0])),
						end=(line_number, index + r.span()[1]),
						line=line,
						file=f.name
					), "Parsing error")
				else:
					yield (t := TokenInfo(
						type=TokenTypes.NEW_LINE,
						string='\n',
						start=(line_number, index),
						end=(line_number, index + 1),
						line=line,
						file=f.name
					))
					if debug: print(t)
				break

class Tokenize:
	def __init__(self, buffer: TextIO, *, debug=False, close=False, parent=None):
		self.buffer: TextIO = buffer
		self.debug: bool = debug
		self.close: bool = close
		self.parent: Optional[TokenInfo] = parent

		self._extend: list[Iterator] = []
		

	def __iter__(self):
		self._tokenize = _tokenize(self.buffer, debug=self.debug)
		return self

	def apply_parent(self, t):
		if self.parent:
			t.parent = self.parent
		return t

	def __next__(self) -> TokenInfo:
		if self._extend:
			try:
				return self.apply_parent(next(self._extend[0]))
			except StopIteration:
				self._extend.pop(0)
		try:
			t = next(self._tokenize)
		except StopIteration:
			if self.close:
				self.buffer.close()
			raise StopIteration
		return self.apply_parent(t)

	def extend(self, tokens):
		self._extend.append(tokens)
