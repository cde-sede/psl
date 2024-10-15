from dataclasses import dataclass
from copy import deepcopy
from enum import Enum, auto

import tree_sitter
from typing import Literal, Any
from typing_extensions import deprecated
from functools import cached_property
import re
from pathlib import Path
import string

VALID_CHARS = string.ascii_letters + string.digits


__all__ = [
	"Type",
	"Types",
	"TokenTypes",
	"Builtins",
	"Syscalls",
	"Operands",
	"Token",
	"Operand",
	"Builtin",
	"Syscall",
	"Include",
	"Cast",
	"Sizeof",
	"Accessor",
	"Definition",
	"Memory",
	"Macro",
	"Proc",
	"Struct",
	"With",
	"Let",
	"FlowControl",
	"While",
	"If",
	"Elif",
	"Else",
	"StackType",
	"Error",
	"Missing",
]

class CONSTANTS:
	CONSTRUCTOR_STRING = '$'
	GETTER_STRING = '~'
	PACK_ALIGN = 8

class Error(Exception):
	args: tuple[tree_sitter.Node, tree_sitter.Node, list[str], str]
class Missing(Exception):
	args: tuple[tree_sitter.Node, tree_sitter.Node, list[str], str]

class Errors:
	class LexerException(Exception):
		args: tuple[tree_sitter.Node, list[str], str]

	class UnknownType(LexerException): pass
	class MissingNode(LexerException): pass
	class InvalidAccessor(LexerException): pass

	class TypeCheckerException(Exception):
		args: tuple[list['Token'], str, Exception | None]


	class ProcedureError(TypeCheckerException): pass

	class UnhandledStackProcedure(ProcedureError): pass
	class ProcedureReturn(ProcedureError): pass
	class InvalidType(TypeCheckerException): pass
	class NotEnoughArguments(TypeCheckerException): pass
	class ReservedKeyword(TypeCheckerException): pass
	class Redefinition(TypeCheckerException): pass
	class NoOverloadMatch(TypeCheckerException): pass
	class NoConstructor(TypeCheckerException): pass
	class NoGetter(TypeCheckerException): pass

	class UnknownField(TypeCheckerException): pass
	class UnknownStructure(TypeCheckerException): pass
	class UnknownWord(TypeCheckerException): pass

	class IfError(TypeCheckerException): pass
	class WhileError(TypeCheckerException): pass


@dataclass
class Type:
	name: str
	size: int
	deref: 'Type | None' = None
	can_deref: bool = False
	is_builtin: bool = False

	def __getitem__(self, key: 'Type'):
		new = deepcopy(self)
		new.deref = deepcopy(key)
		return new

	def __eq__(self, other):
		"""Must match exactly, except for ANY which acts as if the layer is the same for both"""
		assert type(other) == type(self)
		if self.name == other.name or self.name == 'ANY' or other.name == 'ANY':
			if self.deref and other.deref:
				return self.deref == other.deref
			if self.deref or other.deref:
				return False
			return True
		return False

	def __matmul__(self, other):
		"""Must match in size"""
		assert type(other) == type(self)
		if self.size == other.size:
			if self.deref and other.deref:
				return self.deref @ other.deref
			if self.deref or other.deref:
				return False
			return True
		return False

	def __invert__(self):
		return deepcopy(self.deref)

	def __neg__(self):
		new = deepcopy(self)
		new.deref = None
		return new

	def __repr__(self):
		return f"{self.name}{f"[{self.deref!r}]" if self.deref else ''}"

	def __hash__(self):
		return hash(repr(self))

class _IterTypes(type):
	def __iter__(cls):
		return (v for k,v in cls.__dict__.items() if not k.startswith('__'))

class Types(metaclass=_IterTypes):
	ANY		= Type('any', 8, is_builtin=True)

	BOOL	= Type('bool', 1, is_builtin=True)

	CHAR	= Type('char', 1, is_builtin=True)
	SHORT	= Type('short', 2, is_builtin=True)
	INT		= Type('int', 4, is_builtin=True)
	LONG	= Type('long', 8, is_builtin=True)

	UCHAR	= Type('ubool', 1, is_builtin=True)
	USHORT	= Type('ushort', 1, is_builtin=True)
	UINT	= Type('uint', 4, is_builtin=True)
	ULONG	= Type('ulong', 8, is_builtin=True)

	BYTE	= Type('byte', 1, is_builtin=True)
	WORD	= Type('word', 2, is_builtin=True)
	DWORD	= Type('dword', 4, is_builtin=True)
	QWORD	= Type('qword', 8, is_builtin=True)

	PTR		= Type('ptr', 8, can_deref=True, is_builtin=True)

class TokenTypes(Enum):
	WORD       = auto()
	STRING     = auto()
	CHAR       = auto()
	NUMBER     = auto()
	BOOLEANS   = auto()
	OPERAND    = auto()
	KEYWORD    = auto()
	BUILTIN    = auto()
	ACCESSOR   = auto()
	IDENTIFIER = auto()
	SIZEOF     = auto()
	SYSCALLS   = auto()

	INCLUDE    = auto()

	MEMORY     = auto()
	MACRO      = auto()
	STRUCT     = auto()
	ASM        = auto()
	PROC       = auto()
	WITH       = auto()
	LET        = auto()

	CAST       = auto()
	PAIR       = auto()
	FIELD      = auto()

	WHILE      = auto()
	IF         = auto()
	ELIF       = auto()
	ELSE       = auto()

	DO         = auto()
	END        = auto()

class Builtins(Enum):
	DROP      = auto()
	DUP       = auto()
	DUP2      = auto()
	SWAP      = auto()
	OVER      = auto()
	ROT       = auto()
	RROT      = auto()
	EXIT      = auto()
	STORE     = auto()
	LOAD      = auto()

	ARGC      = auto()
	ARGV      = auto()

	DUMP      = auto()
	UDUMP     = auto()
	CDUMP     = auto()
	HEXDUMP   = auto()

class Syscalls(Enum):
	SYSCALL   = auto()
	SYSCALL1  = auto()
	SYSCALL2  = auto()
	SYSCALL3  = auto()
	SYSCALL4  = auto()
	SYSCALL5  = auto()
	SYSCALL6  = auto()

	RSYSCALL  = auto()
	RSYSCALL1 = auto()
	RSYSCALL2 = auto()
	RSYSCALL3 = auto()
	RSYSCALL4 = auto()
	RSYSCALL5 = auto()
	RSYSCALL6 = auto()

class Operands(Enum):
	PLUS       = auto()
	MINUS      = auto()
	MUL        = auto()
	DIVMOD     = auto()
	DIV        = auto()
	MOD        = auto()

	INC        = auto()
	DEC        = auto()

	BLSH       = auto()
	BRSH       = auto()
	BAND       = auto()
	BOR        = auto()
	BXOR       = auto()

	EQ       = auto()
	NE       = auto()
	GT       = auto()
	GE       = auto()
	LT       = auto()
	LE       = auto()

files = []
@dataclass
class Node:
	ts_node: tree_sitter.Node
	file: str | Path | None
	fc: int = -1


	def __post_init__(self):
		if self.file not in files:
			files.append(self.file)
		self.fc = files.index(self.file)

	# Yes I could have used a half decent descriptor
	# or a __getattribute__
	# or a __getattr__
	# but vim magic, sue me
	@property
	def id(self) -> int:
		return self.ts_node.id
	@property
	def kind_id(self) -> int:
		return self.ts_node.kind_id
	@property
	def grammar_id(self) -> int:
		return self.ts_node.grammar_id
	@property
	def grammar_name(self) -> str:
		return self.ts_node.grammar_name
	@property
	def type(self) -> str:
		return self.ts_node.type
	@property
	def is_named(self) -> bool:
		return self.ts_node.is_named
	@property
	def is_extra(self) -> bool:
		return self.ts_node.is_extra
	@property
	def has_changes(self) -> bool:
		return self.ts_node.has_changes
	@property
	def has_error(self) -> bool:
		return self.ts_node.has_error
	@property
	def is_error(self) -> bool:
		return self.ts_node.is_error
	@property
	def parse_state(self) -> int:
		return self.ts_node.parse_state
	@property
	def next_parse_state(self) -> int:
		return self.ts_node.next_parse_state
	@property
	def is_missing(self) -> bool:
		return self.ts_node.is_missing
	@property
	def start_byte(self) -> int:
		return self.ts_node.start_byte
	@property
	def end_byte(self) -> int:
		return self.ts_node.end_byte
	@property
	def byte_range(self) -> tuple[int, int]:
		return self.ts_node.byte_range
	@property
	def range(self) -> tree_sitter.Range:
		return self.ts_node.range
	@property
	def start_point(self) -> tree_sitter.Point:
		return self.ts_node.start_point
	@property
	def end_point(self) -> tree_sitter.Point:
		return self.ts_node.end_point
	@property
	def children(self) -> list[tree_sitter.Node]:
		return self.ts_node.children
	@property
	def child_count(self) -> int:
		return self.ts_node.child_count
	@property
	def named_children(self) -> list[tree_sitter.Node]:
		return self.ts_node.named_children
	@property
	def named_child_count(self) -> int:
		return self.ts_node.named_child_count
	@property
	def parent(self) -> tree_sitter.Node | None:
		return self.ts_node.parent
	@property
	def next_sibling(self) -> tree_sitter.Node | None:
		return self.ts_node.next_sibling
	@property
	def prev_sibling(self) -> tree_sitter.Node | None:
		return self.ts_node.prev_sibling
	@property
	def next_named_sibling(self) -> tree_sitter.Node | None:
		return self.ts_node.next_named_sibling
	@property
	def prev_named_sibling(self) -> tree_sitter.Node | None:
		return self.ts_node.prev_named_sibling
	@property
	def descendant_count(self) -> int:
		return self.ts_node.descendant_count
	@property
	def text(self) -> bytes | None:
		return self.ts_node.text
	def walk(self) -> tree_sitter.TreeCursor:
		return self.ts_node.walk()
	def edit(
			self,
			start_byte: int,
			old_end_byte: int,
			new_end_byte: int,
			start_point: tree_sitter.Point | tuple[int, int],
			old_end_point: tree_sitter.Point | tuple[int, int],
			new_end_point: tree_sitter.Point | tuple[int, int],
			) -> None:
		self.ts_node.edit(start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point)
	def child(self, index: int, /) -> tree_sitter.Node | None:
		return self.ts_node.child(index)
	def named_child(self, index: int, /) -> tree_sitter.Node | None:
		return self.ts_node.named_child(index)
	def child_by_field_id(self, id: int, /) -> tree_sitter.Node | None:
		return self.ts_node.child_by_field_id(id)
	def child_by_field_name(self, name: str, /) -> tree_sitter.Node | None:
		return self.ts_node.child_by_field_name(name)
	def children_by_field_id(self, id: int, /) -> list[tree_sitter.Node]:
		return self.ts_node.children_by_field_id(id)
	def children_by_field_name(self, name: str, /) -> list[tree_sitter.Node]:
		return self.ts_node.children_by_field_name(name)
	def field_name_for_child(self, child_index: int, /) -> str | None:
		return self.ts_node.field_name_for_child(child_index)
	def descendant_for_byte_range(
			self,
			start_byte: int,
			end_byte: int,
			/,
			) -> tree_sitter.Node | None:
		return self.ts_node.descendant_for_byte_range(start_byte, end_byte)
	def named_descendant_for_byte_range(
			self,
			start_byte: int,
			end_byte: int,
			/,
			) -> tree_sitter.Node | None:
		return self.ts_node.named_descendant_for_byte_range(start_byte, end_byte)
	def descendant_for_point_range(
			self,
			start_point: tree_sitter.Point | tuple[int, int],
			end_point: tree_sitter.Point | tuple[int, int],
			/,
			) -> tree_sitter.Node | None:
		return self.ts_node.descendant_for_point_range(start_point, end_point)
	def named_descendant_for_point_range(
			self,
			start_point: tree_sitter.Point | tuple[int, int],
			end_point: tree_sitter.Point | tuple[int, int],
			/,
			) -> tree_sitter.Node | None:
		return self.ts_node.named_descendant_for_point_range(start_point, end_point)
	@deprecated("Use `str()` instead")
	def sexp(self) -> str:
		return str(self.ts_node)
	def __repr__(self) -> str:
		return self.ts_node.__repr__()
	def __str__(self) -> str:
		return self.ts_node.__str__()
	def __eq__(self, other: Any, /) -> bool:
		return self.ts_node.__eq__(other)
	def __ne__(self, other: Any, /) -> bool:
		return self.ts_node.__ne__(other)
	def __hash__(self) -> int:
		return self.ts_node.__hash__()

@dataclass
class Token:
	type: TokenTypes
	text: str
	node: Node

	def __repr__(self):
		return f"{self.type.name}: {self.text}"

	@cached_property
	def label(self) -> str:
		return f"{self.type.name.lower()}_f{self.node.fc}@{self.node.range.start_byte}"

@dataclass
class Operand(Token):
	operand: Operands

	size: int = -1
	def __repr__(self):
		return f"{self.type.name}: {self.operand.name} {self.text}"

@dataclass
class Builtin(Token):
	keyword: Builtins

	def __repr__(self):
		return f"{self.type.name}: {self.keyword.name}"

@dataclass
class Syscall(Token):
	order: Literal[-1, 1]
	nargs: int

	def __repr__(self):
		return f"{self.type.name}: {self.order} {self.nargs}"

@dataclass
class Include(Token):
	file: str
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}: {self.file}\n" +\
		'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

@dataclass
class Cast(Token):
	cast_type: Type

	def __repr__(self):
		return f"{self.type.name}: {self.cast_type}"

@dataclass
class Pair(Token):
	name: Token
	cast: Cast

	def __repr__(self):
		return f"{self.type.name}:\n\t{self.name}\n\t{self.cast}"

@dataclass
class Word(Token):
	data: Any = None

@dataclass
class Sizeof(Token):
	sizeof_type: Type
	size: int

	def __repr__(self):
		return f"{self.type.name}: {self.sizeof_type} -> {self.size}"

@dataclass
class Accessor(Token):
	typ: int
	var: Type
	field: str

	data: Any = None

	def __repr__(self):
		return f"{self.type.name}: {'get' if self.typ else 'set'} {self.var}.{self.field}"

class Definition(Token):
	pass

class Do(Token):
	pass

class End(Token):
	pass

@dataclass
class Memory(Definition):
	name: str
	typ: Type
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}: {self.name}\n" +\
		'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

@dataclass
class Macro(Definition):
	name: str
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}: {self.name}\n" +\
		'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))


@dataclass
class ASM(Token):
	args: list[Cast]
	out: list[Cast]
	body: str

@dataclass
class Proc(Definition):
	name: str
	args: list[Pair]
	out: list[Cast]
	body: list[Token]

	def signature(self) -> list[Cast]:
		return [i.cast for i in self.args]

	@cached_property
	def label(self):
		n = re.sub('_+', '_', ''.join(map(lambda c: c if c in VALID_CHARS else '_', self.name)))
		return f"{n}.f{self.node.fc}@{self.node.range.start_byte}"


	def __repr__(self):
		return f"{self.type.name}: {self.name}\n" +\
				"ARGS:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.args], [])) + '\n' +\
				"OUT:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.out], [])) + '\n' +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))


@dataclass
class Field(Token):
	pair: Pair
	offset: int
	size: int

	def __repr__(self):
		return f"<{self.type.name} {self.pair.name.text} :{self.pair.cast.cast_type} +{self.offset}({self.size})>"



@dataclass
class Struct(Definition):
	name: str
	fields: dict[str, Field]
	methods: dict[str, 'Procs']
	typ: Type

	def get_field(self, name: str) -> Field | None:
		return self.fields.get(name, None)

	def get_offset(self, name: str) -> int:
		field = self.get_field(name)
		if not field:
			raise ValueError(f"Unknown field {name}")
		return field.offset

	def constructor(self) -> 'Procs | None':
		return self.methods.get(CONSTANTS.CONSTRUCTOR_STRING, None)

	def getter(self) -> 'Procs | None':
		return self.methods.get(CONSTANTS.GETTER_STRING, None)

	def __repr__(self):
		return f"{self.type.name}: {self.name}\n" +\
				"FIELDS:\n"+'\n'.join(f"\t{j}" for j in sum([repr(v).split('\n') for k, v in self.fields.items()], []))

@dataclass
class With(Definition):
	variables: list[Pair]
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}\n" +\
				"VARIABLES:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.variables], [])) + '\n' +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

@dataclass
class Let(Definition):
	variables: list[Pair]
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}\n" +\
				"VARIABLES:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.variables], [])) + '\n' +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

class FlowControl(Token):
	pass

@dataclass
class While(Definition):
	condition: list[Token]
	do: Do
	body: list[Token]
	end: End

	def __repr__(self):
		return f"{self.type.name}\n" +\
				"CONDITION:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.condition], [])) + '\n' +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

@dataclass
class If(Definition):
	condition: list[Token]
	do: Do
	body: list[Token]
	elifs: list['Elif']
	else_: 'Else | None'
	end: End

	def __repr__(self):
		return f"{self.type.name}\n" +\
				"CONDITION:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.condition], [])) + '\n' +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], [])) + '\n' +\
				"ELIFS:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.elifs], [])) + '\n' +\
				"ELSE:\n"+'\n'.join(f"\t{j}" for j in repr(self.else_).split('\n'))

@dataclass
class Elif(Definition):
	condition: list[Token]
	do: Do
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}\n" +\
				"CONDITION:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.condition], [])) + '\n' +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

@dataclass
class Else(Definition):
	body: list[Token]

	def __repr__(self):
		return f"{self.type.name}\n" +\
				"BODY:\n"+'\n'.join(f"\t{j}" for j in sum([repr(i).split('\n') for i in self.body], []))

@dataclass
class StackType:
	origin: list[Token]
	typ: Type

	def __repr__(self):
		return f"<{self.origin[-1].text} {self.typ}>"

@dataclass
class Procs:
	name: str
	procs: list[Proc]

	def by_signature(self, signature: list[Type]) -> Proc | None:
		for i in self.procs:
			if len(signature) != len(i.signature()):
				continue
			if all(map(lambda x: x[0] == x[1].cast_type, zip(signature, i.signature()))):
				return i
		return None


