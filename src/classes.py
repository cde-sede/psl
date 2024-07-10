from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TextIO, Any
from dataclasses import dataclass
from enum import Enum
import uuid
from copy import deepcopy
from functools import cached_property

class TypesType(Enum):
	pass

@dataclass
class TokenInfo:
	type: Enum
	string: str
	start: tuple[int, int]
	end: tuple[int, int]
	line: str
	file: str
	parent: 'Optional[TokenInfo]' = None

	def __repr__(self) -> str:
		return f"TokenInfo(type={self.type}, string={self.string!r}, start={self.start!r}, end={self.end!r}, line={self.line!r} file={self.file})"

	def error(self) -> str:
		return f"{self.line}{'': <{self.start[1]}}{'':^<{self.end[1] - self.start[1]}}"

	def copy(self, parent: 'Optional[TokenInfo]'=None):
		return TokenInfo(
			type=self.type,
			string=self.string,
			start=self.start,
			end=self.end ,
			line=self.line,
			file=self.file,
			parent=self.parent if parent is None else parent
		)

@dataclass
class FlowInfo:
	root: 'Token' # OP_IF | OP_WHILE | OP_MACRO
	prev: Optional['Token'] = None
	next: Optional['Token'] = None
	end: Optional['Token'] = None

	haselse: bool = False
	data: Any = None

	def __repr__(self):
		return f"FlowInfo(root={self.root.type}, data={self.data})"


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

	def copy(self, parent: Optional[TokenInfo]=None) -> 'Token':
		return Token(
			value=self.value,
			type=self.type,
			info=self.info.copy(parent) if self.info else (parent if parent else None),
		)


@dataclass
class Type:
	name: str
	_size: int
	parent: 'Type | None' = None

	@property
	def size(self):
		return self.parent.size if self.parent else self._size

	def __getitem__(self, key: 'Type'):
		new = deepcopy(key)
		def f(n):
			if n.parent is not None: f(n.parent)
			else: n.parent = self
		f(new)
		return new

	def __eq__(self, other):
		assert isinstance(other, Type) or other is None
		return ((self.name == other.name
				if other.name != 'ANY' and self.name != 'ANY'
				else True) and (self.parent == other.parent)) if other is not None else False

	def __repr__(self):
		w = f"{self.name}"
		n = self
		while (n := n.parent):
			w = f"{n.name}[{w}]"
		return w

	def __hash__(self):
		return hash(repr(self))


@dataclass
class Symbol:
	type: Any
	data: Any


@dataclass
class Procedure:
	root: Token
	end: Token
	name: str
	args: list[tuple[Token, Type]]
	out: list[tuple[Token, Type]]
	body: list[Token]

class AbstractProgram(ABC):
	class Comment(Exception): ...
	class EndLine(Exception): ...

	instructions: list[Token]
	engine: 'Optional[Engine]'
	path: Path
	pointer: int
	symbols: dict[str, Symbol]
	globals: dict[str, Symbol]
	let_depth: int
	_in_preproc: int
	_position: int
	includes: list[Path]

	@abstractmethod
	def __init__(self, path: str | Path, engine: 'Optional[Engine]'=None, includes: Optional[list[str | Path]]=None):
		...	

	@classmethod
	@abstractmethod
	def frombuffer(cls, buffer: TextIO, path: str | Path, includes: list[str | Path], *, debug=False) -> 'AbstractProgram':
		...	

class Engine(ABC):
	class ExitFromEngine(Exception): pass

	exited: bool = False
	locals: list[dict[str, Any]] = []

	@abstractmethod
	def __init__(self, buffer: TextIO):
		...

	@abstractmethod
	def before(self, program: 'AbstractProgram') -> None:
		...

	@abstractmethod
	def step(self, instruction: Token) -> int:
		...

	@abstractmethod
	def close(self, program):
		...

class Instruction(ABC):
	@property
	def newline(self) -> bool:
		...

	@abstractmethod
	def align(self, largest) -> str:
		...

	@cached_property
	def size(self) -> int:
		...

	def __mod__(self, x: int):
		return f"{self.align(x)}{'\n' if self.newline else ''}"

