from .typechecker import TypeChecker
from functools import cached_property
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import batched, chain

from typing import Any, Literal, cast
import sys

from .classes import (
	Type,
	Types,
	TokenTypes,
	Builtins,
	Syscalls,
	Operands,
	Token,
	Operand,
	Builtin,
	Syscall,
	Word,
	Include,
	Cast,
	Sizeof,
	Accessor,
	Definition,
	Memory,
	Macro,
	ASM as ASMToken,
	Proc,
	Struct,
	Field,
	With,
	Let,
	FlowControl,
	While,
	If,
	Elif,
	Else,
	StackType,
)



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


class Block(Instruction):
	def __init__(self, string, in_, out_):
		self.string = string
		self.in_ = in_
		self.out_ = out_

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t; {self.string}"

	@cached_property
	def size(self) -> int:
		return -1

class Push(Instruction):
	def __init__(self, value):
		self.value = value

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t{'push': <{largest}}{self.value}"

	@cached_property
	def size(self) -> int:
		return 5 + len(self.value) + 1

class Pop(Instruction):
	def __init__(self, value):
		self.value = value

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t{'pop': <{largest}}{self.value}"

	@cached_property
	def size(self) -> int:
		return 4 + len(self.value) + 1

class ASM(Instruction):
	def __init__(self, ins):
		self.ins = ins

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t{self.ins}"

	@cached_property
	def size(self) -> int:
		return len(self.ins) + 1


class ASM1(Instruction):
	def __init__(self, ins, arg0):
		self.ins = ins
		self.arg0 = arg0

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t{self.ins: <{largest}}{self.arg0}"

	@cached_property
	def size(self) -> int:
		return len(self.ins) + 1


class ASM2(Instruction):
	def __init__(self, ins, arg0, arg1):
		self.ins = ins
		self.arg0 = arg0
		self.arg1 = arg1

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t{self.ins: <{largest}}{self.arg0},{self.arg1}"

	@cached_property
	def size(self) -> int:
		return len(self.ins) + 1


class Label(Instruction):
	def __init__(self, name, nl: bool=True):
		self.name = name
		self.nl = nl

	@property
	def newline(self) -> bool:
		return self.nl

	def align(self, largest: int) -> str:
		return f"{f'{self.name}:': <{largest}}"

	@cached_property
	def size(self) -> int:
		return len(self.name) + 1

class Comment(Instruction):
	def __init__(self, text, nl: bool=False):
		self.text = text
		self.nl = nl

	@property
	def newline(self) -> bool:
		return self.nl

	def align(self, largest: int) -> str:
		return f"\t; {self.text}"

	@cached_property
	def size(self) -> int:
		return -1

@dataclass
class LocalVariable:
	name: str
	token: Cast
	id: int
	origin: Literal[TokenTypes.LET, TokenTypes.WITH, TokenTypes.PROC]

	@property
	def label(self):
		return f"var_{self.name}@{self.id}"

@dataclass
class GlobalVariable:
	name: str
	token: Memory
	id: int
	size: int
	can_access: bool

	@property
	def label(self):
		return f"var_{self.name}@{self.id}"


DEBUG_PTR = 0
indent = 0
GLOBALS: list[GlobalVariable] = []

class Compiler:
	INCLUDED = []
	def __init__(self, *, code=None, structs=None, locs=None,
			  macros=None, procs=None, root=False, offset_locals: int=0, with_id: int=0):
		self.code: list[list[Instruction]] = code if code else []

		self.structs: list[Struct] = structs if structs else []
		self.locals: list[dict[str, LocalVariable]] = locs if locs else []
		self.macros: list[Macro] = macros if macros else []
		self.procs: list[Proc] = procs if procs else []

		self.strs: list[Token] = []

		self.labels = 0
		self.num_locals = offset_locals
		self.with_id = with_id

		if root:
			self.start_block()
		else:
			self.block("node", None)

	def start_block(self):
		self.block("INIT", None)
		self.extern('__dump')
		self.extern('__udump')
		self.extern('__cdump')
		self.extern('__hexdump')
		self.extern('__malloc')
		self.extern('__free')
		self.asm1('segment', '.text')
		self.asm1('global', '_start')
		self.block("START", None)
		self.label('_start')
		self.asm2("mov", "qword [ARGS_PTR]", "rsp")

	def block(self, comment: str, token: Token | None) -> None:
		global DEBUG_PTR
		self.code.append([Block(comment, [], [])])
		if token and 1:
			self.label(f"block_{DEBUG_PTR}", nl=True)
			DEBUG_PTR += 1

	def comment(self, text, nl=True):
		self.code[-1].append(Comment(text, nl))

	def label(self, name: str, nl=True, force_unique=False) -> None:
		if force_unique:
			self.code[-1].append(Label(f"{name}_{self.labels}", nl=nl))
		else:
			self.code[-1].append(Label(name, nl=nl))
		self.labels += 1

	def extern(self, fn:str) -> None:
		self.code[0].append(ASM1('extern', fn))

	def asm(self, ins: str) -> None:
		self.code[-1].append(ASM(ins))

	def asm1(self, ins: str, arg: str) -> None:
		self.code[-1].append(ASM1(ins, arg))

	def asm2(self, ins: str, arg1: str, arg2: str) -> None:
		self.code[-1].append(ASM2(ins, arg1, arg2))

	def get_local(self, name: str) -> LocalVariable | None:
		for l in self.locals[::-1]:
			if l.get(name):
				return l[name]
		return None

	def get_global(self, name: str) -> GlobalVariable | None:
		for i in GLOBALS[::-1]:
			if i.name == name and i.can_access:
				return i
		return None

	def get_macro(self, name: str) -> Macro | None:
		for i in self.macros:
			if i.name == name:
				return i
		return None

	def get_proc(self, name: str) -> Proc | None:
		for i in self.procs:
			if i.name == name:
				return i
		return None

	def get_struct(self, name: str) -> Struct | None:
		for i in self.structs:
			if i.name == name:
				return i
		return None

	def get_struct_bytype(self, t: Type) -> Struct | None:
		for i in self.structs:
			if i.typ == t:
				return i
		return None

	def build_proc(self, proc: Proc):
		l = {}
		for i, pair in enumerate(proc.args):
			l[pair.name.text] = LocalVariable(
				pair.name.text,
				pair.cast,
				0x10 + i * 8,
				TokenTypes.PROC
			)
		cc = Compiler(
			structs=self.structs,
			macros=self.macros,
			procs=self.procs,
			locs=[l],
			offset_locals=self.num_locals,
			with_id=self.with_id
		)
		cc.block(f"procedure {proc.name}", proc)
		if proc.args[1::2]:
			cc.comment(f"{' '.join(i.text for i in proc.args[1::2])}",)
		if proc.out:
			cc.comment(f"\t-> {' '.join(i.text for i in proc.out)}")
#		cc.block(f"procedure {proc.name} " +\
#				 f"{' '.join(i.text for i in proc.args[1::2])} " +\
#				 (f"-> {' '.join(i.text for i in proc.out)}" if proc.out else ''), proc)
		cc.label(f"{proc.label}")
		cc.asm1("push", "rbp")
		cc.asm2("mov", "rbp", "rsp")
		cc.run(proc.body)

		for i in range(len(proc.out)):
			cc.asm1("pop", "rax")
			cc.asm2("mov", f"qword [retstack + 0x{i*8:x}]", "rax")

		cc.asm1("pop", "rbp")
		cc.asm1("ret", f"0x{len(proc.args)*8:x}")

		flat = sum(cc.code, [])
		self.num_locals = cc.num_locals
		self.code.insert(1, flat)
		self.strs += cc.strs
		# proc a b c ret d e

		# c
		# b
		# a
		# ptr
		# rbp  <rbp
		# e
		# d
		#      <rsp

		# to return ->
		# for every out
		#	pop rax
		#	mov qword [retstack + i * 8], rax
		#

	def raw_asm(self, source):
		for i in source.split('\n'):
			i = i.strip()
			if not i:
				continue
			if i[-1] == ':':
				if i.rstrip(':').startswith('break_debug'):
					with open(".gdbinit", 'a') as f:
						f.write(f"break {i.rstrip(':')}\n")
				self.label(f"{i.rstrip(':')}")
			else:
				cmd, ws, args = i.partition(' ')
				*args ,= map(str.strip, args.split(','))
				if len(args) == 0: self.asm(cmd.strip())
				elif len(args) == 1: self.asm1(cmd.strip(), *args)
				elif len(args) == 2: self.asm2(cmd.strip(), *args)
				else:
					print(args)

	def call_proc(self, proc: Proc):
		self.asm1("call", f"{proc.label}")
		nargs = len(proc.args)
		offset = (nargs + 3) * 8
		for i in reversed(range(len(proc.out))):
			self.asm2("mov", "rbx", f"qword [retstack + 0x{i*8:x}]")
			self.asm1("push", "rbx")

	def call_cfunction(self, name: str, args: list[Any | None]) -> None:
		registers = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
		for reg, arg in reversed([*zip(registers, args)]):
			if arg is None:
				self.asm1("pop", f"{reg}")
			else:
				if reg != arg:
					self.asm2("mov", f"{reg}", f"{arg}")
		# if len(args) > 6 assume all arguments were already pushed to the stack
		# TODO remove this assumption to handle immediates

		self.asm1("push", "rbp")
		self.asm2("mov", "qword [rsp_align]", "rsp")
		self.asm2("and", "rsp", "-0x10")
		self.asm2("sub", "rsp", "0x10")

		self.asm1("call", f"{name}")

		self.asm2("mov", "rsp", "qword [rsp_align]")
		self.asm1("pop", "rbp")

	def run(self, tokens: list[Token]):
		global indent
		indent += 1
		for index, token in enumerate(tokens):
			#print(f"{'':{'\t'}>{indent}}{token.label}")
			match token:
				case Token(type=TokenTypes.NUMBER) as t:
					self.block("push", t)
					val = int(t.text)
					if val > 2147483647:
						self.asm2("mov", "rax", f"0x{val:x}")
						self.asm1("push", "rax")
					else:
						self.asm1("push", f"0x{val:x}")

				case Token(type=TokenTypes.STRING) as t:
					self.block("push string", t)
					self.asm1("push", f"0x{len(t.text):x}")
					self.asm1("push", f"{t.label}")
					self.strs.append(t)

				case Token(type=TokenTypes.CHAR) as t:
					self.block("push char", t)
					self.asm1("push", f"0x{ord(t.text):x}")

				case Token(type=TokenTypes.BOOLEANS) as t:
					self.block("push bool", t)
					self.asm1("push", f"{1 if t.text == 'true' else 0}")

				case Builtin(keyword=Builtins.DUP) as t:
					self.block('dup', t)
					self.asm1("pop", "rax")
					self.asm1("push", "rax")
					self.asm1("push", "rax")

				case Builtin(keyword=Builtins.DUP2) as t:
					self.block('dup2', t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm1("push", "rbx")
					self.asm1("push", "rax")
					self.asm1("push", "rbx")
					self.asm1("push", "rax")

				case Builtin(keyword=Builtins.SWAP) as t:
					self.block('swap', t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm1("push", "rax")
					self.asm1("push", "rbx")

				case Builtin(keyword=Builtins.OVER) as t:
					self.block('over', t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm1("push", "rbx")
					self.asm1("push", "rax")
					self.asm1("push", "rbx")

				case Builtin(keyword=Builtins.ROT) as t:
					self.block('rot', t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm1("pop", "rcx")
					self.asm1("push", "rbx")
					self.asm1("push", "rax")
					self.asm1("push", "rcx")

				case Builtin(keyword=Builtins.RROT) as t:
					self.block('rrot', t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm1("pop", "rcx")
					self.asm1("push", "rax")
					self.asm1("push", "rcx")
					self.asm1("push", "rbx")

				case Builtin(keyword=Builtins.EXIT) as t:
					for l in self.locals:
						for name, var in l.items():
							if var.origin != TokenTypes.PROC:
								self.call_cfunction("__free", [f"[locals+0x{var.id*8:x}]"])

					self.block('exit', t)
					self.asm1("pop", "rdi")
					self.asm2("mov", "rax", "60")
					self.asm("syscall")

				case Builtin(keyword=Builtins.DROP) as t:
					self.block('exit', t)
					self.asm1("pop", "rax")

				case Builtin(keyword=Builtins.DUMP) as t:
					self.block('dump', t)
					self.call_cfunction("__dump", [None])

				case Builtin(keyword=Builtins.UDUMP) as t:
					self.block('udump', t)
					self.call_cfunction("__udump", [None])

				case Builtin(keyword=Builtins.CDUMP) as t:
					self.block('cdump', t)
					self.call_cfunction("__cdump", [None])

				case Builtin(keyword=Builtins.HEXDUMP) as t:
					self.block('hexdump', t)
					self.call_cfunction("__hexdump", [None])

				case Builtin(keyword=Builtins.ARGC) as t:
					self.block('argc', t)
					self.asm2("mov", "rax", "[ARGS_PTR]")
					self.asm2("mov", "rax", "[rax]")
					self.asm1("push", "rax")

				case Builtin(keyword=Builtins.ARGV) as t:
					self.block('argv', t)
					self.asm2("mov", "rax", "[ARGS_PTR]")
					self.asm2("add", "rax", "0x8")
					self.asm1("push", "rax")

				case Syscall() as t:
					self.block("syscall", t)
					registers = ["rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"][:t.nargs + 1]
					if t.order == -1:
						registers = reversed(registers)
					for reg in registers:
						self.asm1("pop", f"{reg}")
					self.asm("syscall")
					self.asm1("push", "rax")

				case Operand(operand=Operands.PLUS) as t:
					self.block("plus", t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("add", "rbx", "rax")
					self.asm1("push", "rbx")

				case Operand(operand=Operands.MINUS) as t:
					self.block("minus", t)
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("sub", "rbx", "rax")
					self.asm1("push", "rbx")

				case Operand(operand=Operands.MUL) as t:
					self.block("mul", t)
					if t.size == 1:
						self.asm1("pop", "rcx")
						self.asm1("pop", "rax")

						self.asm1("mul", "cl")
						self.asm1("push", "ax")
					elif t.size == 2:
						self.asm1("pop", "rcx")
						self.asm1("pop", "rax")

						self.asm1("mul", "cx")
						self.asm2("shl", "rdx", "16")
						self.asm2("add", "rdx", "rax")
						self.asm1("push", "rax")
					elif t.size == 4:
						self.asm1("pop", "rcx")
						self.asm1("pop", "rax")

						self.asm1("mul", "ecx")
						self.asm2("shl", "rdx", "16")
						self.asm2("add", "rdx", "rax")
						self.asm1("push", "rax")
					elif t.size == 8:
						self.asm1("pop", "rcx")
						self.asm1("pop", "rax")

						self.asm1("mul", "rcx")
						self.asm1("push", "rdx")
						self.asm1("push", "rax")

				case Operand(operand=Operands.DIVMOD) as t:
					self.block("divmod", t)
					self.asm2("xor", "edx", "edx")
					self.asm1("pop", "rsi") # 13
					self.asm1("pop", "rax") # 2
					self.asm1("div", "rsi")
					self.asm1("push", "rax")
					self.asm1("push", "rdx")

				case Operand(operand=Operands.DIV) as t:
					self.block("div", t)
					self.asm2("xor", "edx", "edx")
					self.asm1("pop", "rsi") # 13
					self.asm1("pop", "rax") # 2
					self.asm1("div", "rsi")
					self.asm1("push", "rax")

				case Operand(operand=Operands.MOD) as t:
					self.block("mod", t)
					self.asm2("xor", "edx", "edx")
					self.asm1("pop", "rsi")
					self.asm1("pop", "rax")
					self.asm1("div", "rsi")
					self.asm1("push", "rdx")

				case Operand(operand=Operands.INC) as t:
					self.block("increment", t)
					self.asm1("pop", "rax")
					self.asm1("inc", "rax")
					self.asm1("push", "rax")

				case Operand(operand=Operands.DEC) as t:
					self.block("decrement", t)
					self.asm1("pop", "rax")
					self.asm1("dec", "rax")
					self.asm1("push", "rax")

				case Operand(operand=Operands.BRSH) as t:
					self.block("right shift", t)
					self.asm1("pop", "rcx")
					self.asm1("pop", "rax")
					self.asm2("shr", "rax", "cl")
					self.asm1("push", "rax")

				case Operand(operand=Operands.BLSH) as t:
					self.block("left shift", t)
					self.asm1("pop", "rcx")
					self.asm1("pop", "rax")
					self.asm2("shl", "rax", "cl")
					self.asm1("push", "rax")

				case Operand(operand=Operands.BAND) as t:
					self.block("bitwise and", t)
					self.asm1("pop", "rsi")
					self.asm1("pop", "rax")
					self.asm2("and", "rax", "rsi")
					self.asm1("push", "rax")

				case Operand(operand=Operands.BOR) as t:
					self.block("bitwise bor", t)
					self.asm1("pop", "rsi")
					self.asm1("pop", "rax")
					self.asm2("or", "rax", "rsi")
					self.asm1("push", "rax")

				case Operand(operand=Operands.BXOR) as t:
					self.block("bitwise xor", t)
					self.asm1("pop", "rsi")
					self.asm1("pop", "rax")
					self.asm2("xor", "rax", "rsi")
					self.asm1("push", "rax")

				case Operand(operand=Operands.EQ) as t:
					self.block("eq", t)
					self.asm2("xor", "rcx", "rcx")
					self.asm2("mov", "rdx", "1")
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("cmp", "rbx", "rax")
					self.asm2("cmove", "rcx", "rdx")
					self.asm1("push", "rcx")

				case Operand(operand=Operands.NE) as t:
					self.block("ne", t)
					self.asm2("xor", "rcx", "rcx")
					self.asm2("mov", "rdx", "1")
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("cmp", "rbx", "rax")
					self.asm2("cmovne", "rcx", "rdx")
					self.asm1("push", "rcx")

				case Operand(operand=Operands.GT) as t:
					self.block("gt", t)
					self.asm2("xor", "rcx", "rcx")
					self.asm2("mov", "rdx", "1")
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("cmp", "rbx", "rax")
					self.asm2("cmovg", "rcx", "rdx")
					self.asm1("push", "rcx")

				case Operand(operand=Operands.GE) as t:
					self.block("ge", t)
					self.asm2("xor", "rcx", "rcx")
					self.asm2("mov", "rdx", "1")
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("cmp", "rbx", "rax")
					self.asm2("cmovge", "rcx", "rdx")
					self.asm1("push", "rcx")

				case Operand(operand=Operands.LT) as t:
					self.block("lt", t)
					self.asm2("xor", "rcx", "rcx")
					self.asm2("mov", "rdx", "1")
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("cmp", "rbx", "rax")
					self.asm2("cmovl", "rcx", "rdx")
					self.asm1("push", "rcx")

				case Operand(operand=Operands.LE) as t:
					self.block("le", t)
					self.asm2("xor", "rcx", "rcx")
					self.asm2("mov", "rdx", "1")
					self.asm1("pop", "rax")
					self.asm1("pop", "rbx")
					self.asm2("cmp", "rbx", "rax")
					self.asm2("cmovle", "rcx", "rdx")
					self.asm1("push", "rcx")

				case Struct() as t:
					self.structs.append(t)
					for group in t.methods.values():
						for proc in group.procs:
							self.build_proc(proc)

				case Sizeof() as t:
					self.block(f"sizeof({t.sizeof_type!r})", t)
					self.asm1("push", f"0x{t.size:x}")

				case Word() as t:
					self.block(f'word {t.text}', t)
					if v := self.get_local(t.text):
						if v.origin == TokenTypes.WITH:
							self.asm2("mov", "rax", f"qword [{v.label}]")
							self.asm1("push", "rax")
							# type_size = {1: "byte", 2: "word", 4: "dword", 8: "qword"}[v.token.cast_type.size]
							# register_size = {1: 'al', 2: 'ax', 4: 'eax', 8: 'rax'}[v.token.cast_type.size]
							# self.asm2("mov", "rax", f"qword [locals+0x{v.id*8:x}]")
							# self.asm2("mov", f"{register_size}", f"{type_size} [rax]")
							# self.asm1("push", "rax")
						elif v.origin == TokenTypes.LET:
							self.asm2("mov", "rax", f"qword [locals+0x{v.id*8:x}]")
							self.asm1("push", "rax")
						elif v.origin == TokenTypes.PROC:
							self.asm2("mov", "rax", f"qword [rbp+0x{v.id:x}]")
							self.asm1("push", "rax")
						else:
							raise NotImplementedError()
					elif v := self.get_global(t.text):
						self.asm1("push", f"{v.token.label}")
					elif v := self.get_macro(t.text):
						self.run(v.body)
					elif self.get_proc(t.text):
						self.call_proc(cast(Proc, t.data))
					else:
						raise NotImplementedError(t)

				case Accessor(var=Type(is_builtin=True), field='') as t:
					self.block(f"{'set' if t.typ else 'get'} {t.var}", t)
					if t.typ:					# set
						self.asm1("pop", "rax") # ptr
						self.asm1("pop", "rbx") # value
						if t.var.size == 1:
							self.asm2("mov", f"byte [rax]", "bl")
						elif t.var.size == 2:
							self.asm2("mov", f"word [rax]", "bx")
						elif t.var.size == 4:
							self.asm2("mov", f"dword [rax]", "ebx")
						elif t.var.size == 8:
							self.asm2("mov", f"qword [rax]", "rbx")
						else:
							raise NotImplementedError()
					else:						# get
						self.asm1("pop", "rax") # ptr
						self.asm2("xor", "rbx", "rbx")
						if t.var.size == 1:
							self.asm2("mov", "bl", f"byte [rax]")
						elif t.var.size == 2:
							self.asm2("mov", "bx", f"word [rax]")
						elif t.var.size == 4:
							self.asm2("mov", "ebx", f"dword [rax]")
						elif t.var.size == 8:
							self.asm2("mov", "rbx", f"qword [rax]")
						else:
							raise NotImplementedError()
						self.asm1("push", "rbx")

				case Accessor(var=Type(is_builtin=True)) as t:
					raise NotImplementedError("REPORTING", t)

				case Accessor(var=Type(is_builtin=False), field='') as t:
					self.block(f"{'constructor' if t.typ else 'getter'} {t.var}", t)
					if not isinstance(t.data, Proc):
						raise NotImplementedError("REPORTING", t)
					self.call_proc(cast(Proc, t.data))

				case Accessor(var=Type(is_builtin=False)) as t:
					self.block(f"{'set' if t.typ else 'get'} {t.var}.{t.field}", t)
					field = cast(Field, t.data)
					size, offset = field.size, field.offset
					if not isinstance(offset, int):
						raise NotImplementedError("REPORTING", t)
					if t.typ:					# set
						self.asm1("pop", "rax") # ptr
						self.asm1("pop", "rbx") # value
						if size == 1:
							self.asm2("mov", f"byte [rax+0x{offset:x}]", "bl")
						elif size == 2:
							self.asm2("mov", f"word [rax+0x{offset:x}]", "bx")
						elif size == 4:
							self.asm2("mov", f"dword [rax+0x{offset:x}]", "ebx")
						elif size == 8:
							self.asm2("mov", f"qword [rax+0x{offset:x}]", "rbx")
						else:
							raise NotImplementedError()
					else:						# get
						self.asm1("pop", "rax") # ptr
						self.asm2("xor", "rbx", "rbx")
						if size == 1:
							self.asm2("mov", "bl", f"byte [rax+0x{offset:x}]")
						elif size == 2:
							self.asm2("mov", "bx", f"word [rax+0x{offset:x}]")
						elif size == 4:
							self.asm2("mov", "ebx", f"dword [rax+0x{offset:x}]")
						elif size == 8:
							self.asm2("mov", "rbx", f"qword [rax+0x{offset:x}]")
						else:
							raise NotImplementedError()
						self.asm1("push", "rbx")


				case Cast() as t:
					continue

				case Include() as t:
					# TODO compile in another file object
					if t.file in Compiler.INCLUDED:
						continue
					Compiler.INCLUDED.append(t.file)
					cc = Compiler(
						structs=self.structs,
						procs=self.procs,
						macros=self.macros,
						offset_locals=self.num_locals,
						with_id=self.with_id
					)
					cc.run(t.body)
					self.num_locals = cc.num_locals
					self.macros += cc.macros
					self.structs += cc.structs
					self.procs += cc.procs
					
					flat = sum(cc.code, [])
					self.num_locals = cc.num_locals
					self.code.insert(1, flat)
					self.strs += cc.strs

				case Macro() as t:
					self.macros.append(t)

				case ASMToken() as t:
					self.block("inline ASM", t)
					self.raw_asm(t.body)

				case Proc() as t:
					self.procs.append(t)
					self.build_proc(t)
				
				case If() as t:
					cc = Compiler(
							code=self.code,
							structs=self.structs,
							locs=self.locals,
							macros=self.macros,
							procs=self.procs,
							offset_locals=self.num_locals,
							with_id=self.with_id
					)

					def do_block(do: Token, next_: Token | None, end_: Token):
						cc.block('do', do)
						cc.label(f'{do.label}')
						cc.asm1("pop", "rax")
						cc.asm2("test", "rax", "rax")
						if next_:
							cc.asm1("jz", f"{next_.label}")
						else:
							cc.asm1("jz", f"{end_.label}")

					cc.label(t.label)
					cc.run(t.condition)
					do_block(t.do, t.elifs[0] if t.elifs else t.else_, t.end)

					cc.run(t.body)
					for i, e in enumerate(t.elifs):
						cc.asm1("jmp", f"{t.end.label}")
						cc.label(e.label)
						cc.run(e.condition)
						do_block(e.do, t.elifs[i + 1] if i + 1 < len(t.elifs) else t.else_, t.end)
						cc.run(e.body)

					if t.else_:
						cc.asm1("jmp", f"{t.end.label}")
						cc.label(f"{t.else_.label}")
						cc.run(t.else_.body)
					cc.label(f"{t.end.label}")
					self.strs += cc.strs

				case While() as t:
					cc = Compiler(
							code=self.code,
							structs=self.structs,
							locs=self.locals,
							macros=self.macros,
							procs=self.procs,
							offset_locals=self.num_locals,
							with_id=self.with_id
					)
					cc.label(t.label)
					cc.run(t.condition)
					cc.block('do', t.do)
					cc.label(f"{t.do.label}")
					cc.asm1("pop", "rax")
					cc.asm2("test", "rax", "rax")
					cc.asm1("jz", f"{t.end.label}")
					cc.run(t.body)
					cc.asm1("jmp", f"{t.label}")
					cc.label(f"{t.end.label}")
					self.strs += cc.strs

				case Memory() as t:
					s = []
					simple_simulate(s, t.body, self.structs, self.macros)
					if len(s) != 1:
						raise NotImplementedError("REPORTING")
					GLOBALS.append(GlobalVariable(
							name=t.name,
							token=t,
							id=len(GLOBALS),
							size=s[0] * t.typ.size,
							can_access=True
					))

				case With() as t:
					self.block(f"with {' '.join(i.text for i in t.variables[::2])}", t)
					l = {}
					self.with_id += 1
					for pair in t.variables:
						var = LocalVariable(
							pair.name.text,
							pair.cast,
							self.with_id,
							TokenTypes.WITH
						)

						self.asm1("pop", "rbx")
						self.asm2("mov", f"qword [{var.label}]", "rbx")

						l[pair.name.text] = var

						# self.call_cfunction("__malloc", ["8"])
						# self.asm2("mov", f"qword [locals + 0x{self.num_locals * 8:x}]", "rax")
						# self.asm1("pop", "rbx")
						# self.asm2("mov", "qword [rax]", "rbx")
						# l[pair.name.text] = LocalVariable(
						# 	pair.name.text,
						# 	pair.cast,
						# 	self.num_locals,
						# 	TokenTypes.WITH
						# )
						# self.num_locals += 1
					self.locals.append(l)
					cc = Compiler(
							code=self.code,
							structs=self.structs,
							locs=self.locals,
							macros=self.macros,
							procs=self.procs,
							offset_locals=self.num_locals,
							with_id=self.with_id
					)
					cc.run(t.body)
					self.num_locals = max(self.num_locals, cc.num_locals)
					for name, var in self.locals.pop().items():
						if var.origin != TokenTypes.PROC:
							self.asm2("mov", f"qword [{var.label}]", "0")
							GLOBALS.append(GlobalVariable(
									name=var.name,
									token=cast(Memory, var),
									id=var.id,
									size=var.token.cast_type.size,
									can_access=False
							))
#							self.call_cfunction("__free", [f"[locals+0x{var.id*8:x}]"])
					self.strs += cc.strs

				case Let() as t:
					self.block(f"let {' '.join(i.text for i in t.variables[::2])}", t)
					l = {}
					for pair in t.variables:
						sizeof = pair.cast.cast_type.size
						self.asm1("pop", "rdi")
						self.asm2("shl", "rdi", "3")
						self.call_cfunction("__malloc", ['rdi'])
						self.asm2("mov", f"qword [locals + 0x{self.num_locals * 8:x}]", "rax")
						
						self.asm2("mov", f"qword [rax]", "0")
						l[pair.name.text] = LocalVariable(
							pair.name.text,
							pair.cast,
							self.num_locals,
							TokenTypes.LET
						)
						self.num_locals += 1
					self.locals.append(l)
					cc = Compiler(
							code=self.code,
							structs=self.structs,
							locs=self.locals,
							macros=self.macros,
							procs=self.procs,
							offset_locals=self.num_locals,
							with_id=self.with_id
					)
					cc.run(t.body)
					self.num_locals = max(self.num_locals, cc.num_locals)
					for name, var in self.locals.pop().items():
						if var.origin != TokenTypes.PROC:
							self.call_cfunction("__free", [f"[locals+0x{var.id*8:x}]"])
					self.strs += cc.strs

				case _:
					raise NotImplementedError(token)
		indent -= 1

	def close(self, buffer):
		self.block("DATA", None)
		self.asm1("segment", ".data")
		for index, s in enumerate(self.strs):
			self.label(f"{s.label}", nl=False)
			self.asm1("db", ','.join(map(hex, s.text.encode('utf8'))) + f'{',' if s.text else ''}0x0')

		self.label("rsp_align", nl=False)
		self.asm1("dq", "0x0")

		
#		if self.num_locals:
#			self.label("locals", nl=False)
#			self.asm1("dq", f"{','.join('0x0' for i in range(self.num_locals))}")
		
		self.block("MEMORY", None)
			

		self.asm1("segment", ".bss")
		self.label(f"ARGS_PTR", nl=False); self.asm1("resq", "1")
		self.label(f"retstack", nl=False); self.asm1("resq", "128")

		if self.num_locals:
			self.label("locals", nl=False)
			self.asm1("resq", f"{self.num_locals}")


		for i in GLOBALS:
			self.label(f"{i.token.label}", nl=False)
			self.asm1("resb", f"{i.size}")

		align = max([j.size for i in self.code for j in i])
		for i in self.code:
			for j in i:
				buffer.write(j % align)
			buffer.write('\n')

def get_type(s, structs: list[Struct]) -> Type:
	t, l = {
		i.name: (i, len(i.name)) for i in chain(
			Types,
			(j.typ for j in structs)
		)
	}.get(s.rstrip('*'), [None, None])

	if t is None or l is None:
		raise ValueError(s)

	while s[l:] and s[l:][0] == '*':
		t = Types.PTR[t]
		l += 1
	return t

def simple_simulate(stack: list[int], tokens: list[Token],
					structs: list[Struct],
					macros: list[Macro]):
	for i in tokens:
		match i:
			case Token(type=TokenTypes.NUMBER) as t:
				stack.append(int(t.text))

			case Operand(operand=Operands.PLUS) as t:
				b = stack.pop()
				a = stack.pop()
				stack.append(b + a)

			case Operand(operand=Operands.MINUS) as t:
				b = stack.pop()
				a = stack.pop()
				stack.append(b - a)

			case Operand(operand=Operands.MUL) as t:
				b = stack.pop()
				a = stack.pop()
				stack.append(b * a)

			case Operand(operand=Operands.DIV) as t:
				b = stack.pop()
				a = stack.pop()
				stack.append(b // a)

			case Operand(operand=Operands.DIV) as t:
				b = stack.pop()
				a = stack.pop()
				stack.append(b // a)

			case Sizeof() as t:
				t = get_type(t.text[1:], structs)
				if t is None:
					raise NotImplementedError("REPORTING")
				stack.append(t.size)

			case Word() as t:
				for i in macros:
					if i.name == t.text:
						simple_simulate(stack, i.body, structs, macros)
						continue
				raise NotImplementedError("REPORTING")


			case _:
				raise NotImplementedError("REPORTING")
