from abc import ABC, abstractmethod
from typing import Any, TextIO, BinaryIO, Generator, Iterator
from functools import cached_property
from queue import LifoQueue, Empty
from itertools import chain, takewhile


import ctypes
import sys

from .classes import (
	Token, Procedure, Type, Instruction, Engine
)

from .lexer import (
	TokenInfo,
	TokenTypes,

	TypesType,
	OpTypes,
	PreprocTypes,
	Intrinsics,
	Operands,
	FlowControl,

)

from .typechecker import (
	TypeChecker,
	run_single,
	Types,
)

from .errors import (
	TypeWarning,
	Stopped,
	Reporting,
	InvalidSyntax,
)

from .error_trace import (
	warn, trace,
)

STR_CAPACITY    = 640_000
MEM_CAPACITY    = 64_000
ARGV_CAPACITY   = 64_000
LOCALS_CAPACITY = 512    # meaning it can store up to 512 vars, should be enough but honestly no idea



class Block(Instruction):
	def __init__(self, string):
		self.string = string

	@property
	def newline(self) -> bool:
		return True

	def align(self, largest: int) -> str:
		return f"\t; {self.string}"

	@cached_property
	def size(self) -> int:
		return -1

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

class Compiler(Engine):
	def __init__(self, buffer: TextIO):
		self.buffer: TextIO = buffer
		self._code: list[list[Instruction]] = []
		self.strs = []
		self.labels = 0

		self.locals: list[dict[str, tuple[int, TypesType]]] = []
		self.globals: dict[str, tuple[int, Token]] = {}
		self.procs: dict[str, Procedure] = {}
		self.num_locals: int = 0

		self.checktype, self.checker = TypeChecker()
		self._state = 0

		self.block("SLANG COMPILED PROGRAM", None)

		self.asm1('extern', '__dump')
		self.asm1('extern', '__udump')
		self.asm1('extern', '__hexdump')
		self.asm1('extern', '__cdump')
		self.asm1('extern', '__malloc')
		self.asm1('extern', '__free')
		self.asm1('segment', '.text')
		self.asm1('global', '_start')
		self.label('\n_start')
		self.asm2("mov", "qword [ARGS_PTR]", "rsp")


	def before(self, program) -> None:
		for name, symbol in program.globals.items():
			interp = Interpreter(sys.stdout.buffer)
			token: Token = symbol.data
			for op in token.value.data[1:]:
				if op.type not in [OpTypes.OP_PUSH, Operands.OP_PLUS, Operands.OP_MINUS, Operands.OP_MUL, Operands.OP_DIV, FlowControl.OP_LABEL]:
					raise InvalidSyntax(op.info, "Only trivial intrinsics and substitutions are allowed inside memory block", Reporting(token.info, ''))
				interp.step(op)
			size = interp.pop()
			if not interp.queue.empty():
				raise InvalidSyntax(token.info, "Memory accepts only 1 argument")
			self.globals[name] = (size, token)

	def block(self, comment: str, token: Token | None) -> None:
		self._code.append([Block(comment)])
		if token and 1:
			self.label(f"block_{token.type.name}_{len(self._code)}", nl=True)

	def label(self, name: str, nl=True, force_unique=False) -> None:
		if force_unique:
			self._code[-1].append(Label(f"{name}_{self.labels}", nl=nl))
		else:
			self._code[-1].append(Label(name, nl=nl))
		self.labels += 1

	def asm(self, ins: str) -> None:
		self._code[-1].append(ASM(ins))

	def asm1(self, ins: str, arg: str) -> None:
		self._code[-1].append(ASM1(ins, arg))

	def asm2(self, ins: str, arg1: str, arg2: str) -> None:
		self._code[-1].append(ASM2(ins, arg1, arg2))

	def call_cfunction(self, name: str, args: list[Any | None]) -> None:
		"""  x86 arg registers
		arg0 (%rdi)	arg1 (%rsi)	arg2 (%rdx)	arg3 (%r10)	arg4 (%r8)	arg5 (%r9)
		"""

		registers = ["rdi", "rsi", "rdx", "r10", "r8", "r9"]
		for reg, arg in reversed([*zip(registers, args)]):
			if arg is None:
				self.asm1("pop", f"{reg}")
			else:
				self.asm2("mov", f"{reg}", f"{arg}")
		# if len(args) > 6 assume all arguments were already pushed to the stack
		# TODO remove this assumption to handle immediates

		self.asm1("push", "rbp")
		self.asm2("mov", "qword [rsp_align]", "rsp")
		self.asm2("and", "rsp", "-0x10")
		self.asm2("sub", "rsp", "8")

		self.asm1("call", f"{name}")

		self.asm2("mov", "rsp", "qword [rsp_align]")
		self.asm1("pop", "rbp")


	def step(self, instruction: Token) -> int:
		try:
			self.checktype.send(instruction)
		except TypeWarning as e:
			self._state |= 0b1
			warn(e)
		except TypeError as e:
			self._state |= 0b10
			trace(e)
		match instruction:
			case Token(type=FlowControl.OP_LABEL, value=val):
				pass
			#	self.block("label", instruction)
			#				self.label(val, force_unique=True)

			case Token(type=OpTypes.OP_PUSH, value=val):
				self.block("push", instruction)
				if val > 2147483647:
					self.asm2("mov", "rax", f"0x{val:x}")
					self.asm1("push", "rax")
				else:
					self.asm1("push", f"0x{val:x}")

			case Token(type=OpTypes.OP_CHAR, value=val):
				self.block("push char", instruction)
				self.asm1("push", f"0x{val:x}")

			case Token(type=OpTypes.OP_BOOL, value=val):
				self.block("push bool", instruction)
				self.asm1("push", f"{int(val)}")

			case Token(type=OpTypes.OP_STRING, value=val):
				self.block("push string", instruction)
				self.asm2("mov", "rax", f"{len(val)}")
				self.asm1("push", "rax")
				self.asm1("push", f"STR_{len(self.strs)}")
				self.strs.append(val)

			case Token(type=Intrinsics.OP_DROP, value=val):
				self.block("pop", instruction)
				self.asm1("pop", "rax")

			case Token(type=Intrinsics.OP_DUP, value=val):
				self.block("dup", instruction)
				self.asm1("pop", "rax")
				self.asm1("push", "rax")
				self.asm1("push", "rax")

			case Token(type=Intrinsics.OP_DUP2, value=val):
				self.block("dup", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")

			case Token(type=Intrinsics.OP_SWAP, value=val):
				self.block("swap", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm1("push", "rax")
				self.asm1("push", "rbx")

			case Token(type=Intrinsics.OP_OVER, value=val):
				self.block("over", instruction)
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm1("push", "rax")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")

			case Token(type=Intrinsics.OP_ROT, value=val):
				self.block("rot", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm1("pop", "rcx")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")
				self.asm1("push", "rcx")

			case Token(type=Intrinsics.OP_RROT, value=val):
				self.block("rot", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm1("pop", "rcx")
				self.asm1("push", "rax")
				self.asm1("push", "rcx")
				self.asm1("push", "rbx")

			case Token(type=Operands.OP_PLUS, value=val):
				self.block("plus", instruction)
				self.asm1("pop", "rax") # INT
				self.asm1("pop", "rbx") # INT
				self.asm2("add", "rax", "rbx")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_MINUS, value=val):
				self.block("minus", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("sub", "rbx", "rax")
				self.asm1("push", "rbx")

			case Token(type=Operands.OP_MUL, value=val):
				self.block("mul", instruction)
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm1("mul", "rcx")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_DIV, value=val):
				self.block("div", instruction)
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_MOD, value=val):
				self.block("div", instruction)
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rdx")

			case Token(type=Operands.OP_DIVMOD, value=val):
				self.block("divmod", instruction)
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rax")
				self.asm1("push", "rdx")

			case Token(type=Operands.OP_INCREMENT, value=val):
				self.block("increment", instruction)
				self.asm1("pop", "rax")
				self.asm1("inc", "rax")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_DECREMENT, value=val):
				self.block("decrement", instruction)
				self.asm1("pop", "rax")
				self.asm1("dec", "rax")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_BLSH, value=val):
				self.block("bitwise shift left", instruction)
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm2("shl", "rax", "cl")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_BRSH, value=val):
				self.block("bitwise shift right", instruction)
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm2("shr", "rax", "cl")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_BAND, value=val):
				self.block("bitwise and", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm2("and", "rax", "rsi")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_BOR, value=val):
				self.block("bitwise or", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm2("or", "rax", "rsi")
				self.asm1("push", "rax")

			case Token(type=Operands.OP_EQ, value=val):
				self.block("eq", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmove", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=Operands.OP_NE, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovne", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=Operands.OP_GT, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovg", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=Operands.OP_GE, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovge", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=Operands.OP_LT, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovl", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=Operands.OP_LE, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovle", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=OpTypes.OP_DUMP, value=val):
				self.block("dump", instruction)
				self.call_cfunction("__dump", [None])

			case Token(type=OpTypes.OP_UDUMP, value=val):
				self.block("udump", instruction)
				self.call_cfunction("__udump", [None])

			case Token(type=OpTypes.OP_CDUMP, value=val):
				self.block("cdump", instruction)
				self.call_cfunction("__cdump", [None])

			case Token(type=OpTypes.OP_HEXDUMP, value=val):
				self.block("hexdump", instruction)
				self.call_cfunction("__hexdump", [None])

			case Token(type=OpTypes.OP_SYSCALL, value=val):
				self.block("syscall", instruction)
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_RSYSCALL1, value=val):
				self.block("rsyscall1", instruction)
				self.asm1("pop", "rdi")
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_RSYSCALL2, value=val):
				self.block("rsyscall2", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_RSYSCALL3, value=val):
				self.block("rsyscall3", instruction)
				self.asm1("pop", "rdx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_RSYSCALL4, value=val):
				self.block("rsyscall4", instruction)
				self.asm1("pop", "r10")
				self.asm1("pop", "rdx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_RSYSCALL5, value=val):
				self.block("rsyscall5", instruction)
				self.asm1("pop", "r8")
				self.asm1("pop", "r10")
				self.asm1("pop", "rdx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_RSYSCALL6, value=val):
				self.block("rsyscall5", instruction)
				self.asm1("pop", "r9")
				self.asm1("pop", "r8")
				self.asm1("pop", "r10")
				self.asm1("pop", "rdx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rax")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SYSCALL1, value=val):
				self.block("syscall1", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rdi")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SYSCALL2, value=val):
				self.block("syscall2", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rsi")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SYSCALL3, value=val):
				self.block("syscall3", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdx")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SYSCALL4, value=val):
				self.block("syscall4", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdx")
				self.asm1("pop", "r10")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SYSCALL5, value=val):
				self.block("syscall5", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdx")
				self.asm1("pop", "r10")
				self.asm1("pop", "r8")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SYSCALL6, value=val):
				self.block("syscall6", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rdi")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rdx")
				self.asm1("pop", "r10")
				self.asm1("pop", "r8")
				self.asm1("pop", "r9")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_EXIT, value=val):
				self.block("EXIT", instruction)
				for l in self.locals[::-1]:
					for offset, typ in l.values():
						self.call_cfunction("__free", [f"[locals+0x{offset*8:x}]"])
#						self.asm2("mov", "rdi", f"[locals+{offset*8}]")
#						self.asm1("call", "__free")
				self.asm1("pop", "rdi")
				self.asm2("mov", "rax", "60")
				self.asm("syscall")
				self.asm1("push", "rax")

				self.exited = True
#				raise self.ExitFromEngine(0)

			case Token(type=FlowControl.OP_IF, value=val):
				self.block("IF", instruction)
				self.label(instruction.label())
			
			case Token(type=FlowControl.OP_ELIF, value=val):
				self.block("ELIF", instruction)
				self.asm1("jmp", f"{val.end.label()}")
				self.label(instruction.label())

			case Token(type=FlowControl.OP_ELSE, value=val):
				self.block("ELSE", instruction)
				self.asm1("jmp", f"{val.end.label()}")
				self.label(instruction.label())


			case Token(type=FlowControl.OP_WHILE, value=val):
				self.block("WHILE", instruction)
				self.label(instruction.label())

			case Token(type=FlowControl.OP_DO, value=val):
				if val.root.type in [FlowControl.OP_WHILE, FlowControl.OP_IF, FlowControl.OP_ELIF]:
					self.block("DO", instruction)
					self.label(instruction.label())
					self.asm1("pop", "rax")
					self.asm2("test", "rax", "rax")
					if val.next:
						self.asm1("jz", f"{val.next.label()}")
					else:
						self.asm1("jz", f"{val.end.label()}")

			case Token(type=FlowControl.OP_END, value=val):
				if val.root.type in [PreprocTypes.PROC,]:
					self.block("END", instruction)
					proc = val.root.value
					nout = len(proc.out)
					nargs = len(proc.args)
					self.asm2("add", "rsp", f"0x{nout * 8:x}")
					self.asm1("pop", "rbp")
#					self.asm2("mov", "rax", "rbp")
					self.asm1("ret", f"0x{nargs * 8:x}")
					self.label(instruction.label())
					self.locals.pop()
				else:
					self.block("END", instruction)
					if val.root.type in [FlowControl.OP_WHILE,]:
						self.asm1("jmp", f"{val.root.label()}")
					if val.root.type in [FlowControl.OP_WITH, FlowControl.OP_LET]:
						for offset, typ in self.locals.pop().values():
							self.call_cfunction("__free", [f"[locals+0x{offset*8:x}]"])
#							self.asm2("mov", "rdi", f"[locals+0x{offset * 8:x}]")
#							self.asm1("call", "__free")
					self.label(instruction.label())

			case Token(type=FlowControl.OP_LET, value=val):
				self.block(f"let {' '.join(i.value for i in val.data)}", instruction)
				l = {}
				for var in val.data:
					#self.asm1("pop", "rdi")
					#self.asm1("call", "__malloc")
					self.call_cfunction("__malloc", [None])

					self.asm2("mov", f"qword [locals+0x{self.num_locals * 8:x}]", "rax")
					self.asm2("mov", "qword [rax]", "0")
					l[var.value] = (self.num_locals, FlowControl.OP_LET)
					self.num_locals += 1
				self.locals.append(l)

			case Token(type=FlowControl.OP_WITH, value=val):
				self.block(f"with {' '.join(i.value for i in val.data)}", instruction)
				l = {}
				for var in val.data:
					#self.asm2("mov", "rdi", "8") # f"{var.size}"?
					#self.asm1("call", "__malloc")
					self.call_cfunction("__malloc", ["8"])

					self.asm2("mov", f"qword [locals+0x{self.num_locals * 8:x}]", "rax")
					self.asm1("pop", "rbx")
					self.asm2("mov", "qword [rax]", "rbx")
					l[var.value] = (self.num_locals, FlowControl.OP_WITH)
					self.num_locals += 1
				self.locals.append(l)

			case Token(type=Intrinsics.OP_ARGC, value=val):
				self.block("argc", instruction)
				self.asm2("mov", "rax", "[ARGS_PTR]")
				self.asm2("mov", "rax", "[rax]")
				self.asm1("push", "rax")

			case Token(type=Intrinsics.OP_ARGV, value=val):
				self.block("argc", instruction)
				self.asm2("mov", "rax", "[ARGS_PTR]")
				self.asm2("add", "rax", "8")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_STORE, value=val):
				self.block("store", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("mov", "byte [rax]", "bl")

			case Token(type=OpTypes.OP_LOAD, value=val):
				self.block("load", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "bl", "byte [rax]")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_STORE16, value=val):
				self.block("store 16", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("mov", "word [rax]", "bx")

			case Token(type=OpTypes.OP_LOAD16, value=val):
				self.block("load 16", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "bx", "word [rax]")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_STORE32, value=val):
				self.block("store 32", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("mov", "dword [rax]", "ebx")

			case Token(type=OpTypes.OP_LOAD32, value=val):
				self.block("load 32", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "ebx", "dword [rax]")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_STORE64, value=val):
				self.block("store 64", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("mov", "[rax]", "rbx")

			case Token(type=OpTypes.OP_LOAD64, value=val):
				self.block("load 64", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "rbx", "[rax]")
				self.asm1("push", "rbx")

			case Token(type=PreprocTypes.MEMORY, value=val):
				name = val.data[0].value
				if name in self.globals:
					size, token = self.globals[name]
					self.asm1("push", f"{token.label()}")
				else:
					raise ValueError("Should have been caught by parser")

			case Token(type=OpTypes.OP_WORD, value=val):
				for d in reversed(self.locals):
					if d.get(val, None) is not None:
						self.block(f"var {val}", instruction)
						offset, typ = d[val]
						if typ == FlowControl.OP_WITH:
							self.asm2("mov", "rax", f"[locals+0x{offset * 8:x}]")
							self.asm2("mov", "rbx", "qword [rax]")
							self.asm1("push", "rbx")
						elif typ == FlowControl.OP_LET:
							self.asm2("mov", "rax", f"[locals+0x{offset * 8:x}]")
							self.asm1("push", "rax")
						elif typ == PreprocTypes.PROC:
							self.asm2("mov", "rax", f"qword [rbp+0x{offset:x}]")
							self.asm1("push", "rax")
						break
				else:
					if val in self.procs:
						self.block(f"CALL {val}", instruction)
						proc = self.procs[val]
						self.asm1("call", f"{proc.root.label()}")
						nout = len(proc.out)
						nargs = len(proc.args)
						offset = (nargs + 3) * 8
						for i in range(nout):
							self.asm2("mov", "rbx", f"qword [rsp-0x{offset:x}]")
							self.asm1("push", "rbx")
					else:
						raise ValueError("Should have been caught by the type checker")

			case Token(type=PreprocTypes.CALL, value=val):
				self.block(f"CALL {val}", instruction)
				proc = self.procs[val]
				self.asm1("call", f"{proc.root.label()}")
				nout = len(proc.out)
				nargs = len(proc.args)
				offset = (nargs + 3) * 8
				for i in range(nout):
					self.asm2("mov", "rbx", f"qword [rsp-0x{offset:x}]")
					self.asm1("push", "rbx")

			case Token(type=PreprocTypes.PROC, value=val):
				self.procs[val.name] = val
				self.asm1("jmp", f"{val.end.label()}")
				self.label(instruction.label())
				self.asm1("push", "rbp")
				self.asm2("mov", "rbp", "rsp")
				l = {}
				for i, (tok, typ) in enumerate(val.args):
					l[tok.value] = (0x10 + i * 8, PreprocTypes.PROC)
				self.locals.append(l)
#				for i in range(len(val.next.value.data[1:])):
#					self.asm2("mov", "rax", f"qword [rbp+0x{0x10 + i * 8:x}]")
#					self.asm1("push", "rax")

			case Token(type=PreprocTypes.CAST):
				pass

			case _:
				raise RuntimeError(NotImplemented, instruction)

		return 0

	def close(self, program):
		try:
			self.checktype.send(None)
		except TypeWarning as e:
			self._state |= 0b1
			warn(e)
		except TypeError as e:
			self._state |= 0b10
			trace(e)
		if self._state & 0b10:
			raise Stopped()


		self.block("DATA", None)
		self.asm1("segment", ".data")
		for index, s in enumerate(self.strs):
			self.label(f"STR_{index}", nl=False)
			self.asm1("db", ','.join(map(hex, s.encode('utf8'))) + ',0x0')

		self.label("rsp_align", nl=False)
		self.asm1("dq", "1")
		
		self.block("MEMORY", None)

		self.asm1("segment", ".bss")
		self.label(f"ARGS_PTR", nl=False); self.asm1("resq", "1")
		
		for name, (size, token) in self.globals.items():
			self.label(token.label(), nl=False)
			self.asm1("resb", f"{size}")

		if self.num_locals:
			self.label("locals", nl=False)
			self.asm1("resq", f"{self.num_locals}")

		self.buffer.write("BITS 64\n")
		align = max([j.size for i in self._code for j in i])
		for i in self._code:
			for j in i:
				self.buffer.write(j % align)
			self.buffer.write('\n')

class Interpreter(Engine):
	def __init__(self, buffer: BinaryIO):
		self.queue = LifoQueue(-1)
		self.memory = bytearray(1 + STR_CAPACITY + ARGV_CAPACITY + MEM_CAPACITY)

		self.str_ptr = 1
		self.argv_ptr = 1 + STR_CAPACITY
		self.argc = 0

		self.strs: dict[Token, tuple[int, int]] = {}

		self.last_case = -1
		self.type_stack = []

		self.fds: dict[int, BinaryIO] = {
			0: sys.stdin.buffer,
			1: buffer,
			2: sys.stdout.buffer,
		}

	def setargv(self, av):
		for i in av:
			l, str_ptr = self.set_string(i)

			p = self.argv_ptr + self.argc*8

			assert p + 8 < 1 + STR_CAPACITY + ARGV_CAPACITY, "Argv overflow"

			self.memory[p:p+8] = str_ptr.to_bytes(8, byteorder='little')
			self.argc += 1

	def set_string(self, s: str) -> tuple[int, int]:
		value = s.encode('utf8')
		n = len(value)
		self.memory[self.str_ptr:self.str_ptr+n] = value
		self.memory[self.str_ptr+n+1] = 0
		p = self.str_ptr

		self.str_ptr += n + 1

		assert self.str_ptr < 1 + STR_CAPACITY, "String overflow"
		return n + 1, p

	def push(self, v: Any):
		self.queue.put(v)

	def pop(self) -> Any:
		return self.queue.get_nowait()

	def before(self, program) -> None:
		checktype, _ = TypeChecker()
		state = 0
		for i in chain(program.instructions, [None]):
			try:
				checktype.send(i)
			except TypeWarning as e:
				state |= 0b1
				warn(e)
			except TypeError as e:
				state |= 0b10
				trace(e)
		if state & 0b10:
			raise Stopped()

	def step(self, instruction: Token):
		self.last_case, self.type_stack, last_type = run_single(instruction, self.type_stack)
		match instruction:
			case Token(type=OpTypes.OP_PUSH, value=val):
				self.queue.put(val)

			case Token(type=OpTypes.OP_CHAR, value=val):
				self.queue.put(val)

			case Token(type=OpTypes.OP_BOOL, value=val):
				self.queue.put(int(val))

			case Token(type=OpTypes.OP_STRING, value=val):
				if instruction in self.strs:
					l, p = self.strs[instruction]
				else:
					l, p = self.set_string(val)
					self.strs[instruction] = (l, p)
				self.push(l)
				self.push(p)

			case Token(type=Intrinsics.OP_DROP, value=val):
				self.pop()

			case Token(type=Intrinsics.OP_DUP, value=val):
				a = self.pop()
				self.push(a)
				self.push(a)

			case Token(type=Intrinsics.OP_DUP2, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b)
				self.push(a)
				self.push(b)
				self.push(a)

			case Token(type=Intrinsics.OP_SWAP, value=val):
				a = self.pop()
				b = self.pop()
				self.push(a)
				self.push(b)

			case Token(type=Intrinsics.OP_OVER, value=val):
				b = self.pop()
				a = self.pop()
				self.push(a)
				self.push(b)
				self.push(a)

			case Token(type=Intrinsics.OP_ROT, value=val):
				a = self.pop()
				b = self.pop()
				c = self.pop()
				self.push(b)
				self.push(a)
				self.push(c)

			case Token(type=Intrinsics.OP_RROT, value=val):
				a = self.pop()
				b = self.pop()
				c = self.pop()
				self.push(a)
				self.push(c)
				self.push(b)

			case Token(type=Operands.OP_PLUS, value=val):
				a = self.pop()
				b = self.pop()
#				if self.last_case == 1:
#					self.push(a * 8 + b)
#				elif self.last_case == 2:
#					self.push(a + b * 8)
#				else:
#					self.push(a + b)
				self.push(a + b)

			case Token(type=Operands.OP_MINUS, value=val):
				a = self.pop()
				b = self.pop()
#				if self.last_case == 1:
#					self.push(b - a * 8)
#				elif self.last_case == 2:
#					self.push((b - a) // 8)
#				else:
#					self.push(b - a)
				self.push(b - a)

			case Token(type=Operands.OP_MUL, value=val):
				self.push(self.pop() * self.pop())

			case Token(type=Operands.OP_DIV, value=val):
				b = self.pop()
				a = self.pop()
				self.push(a // b)

			case Token(type=Operands.OP_DIVMOD, value=val):
				b = self.pop()
				a = self.pop()
				self.push(a // b)
				self.push(a % b)

			case Token(type=Operands.OP_INCREMENT, value=val):
				a = self.pop()
#				if self.last_case == 1:
#					self.push(a + 8)
#				else:
#					self.push(a + 1)
				self.push(a + 1)

			case Token(type=Operands.OP_DECREMENT, value=val):
				a = self.pop()
#				if self.last_case == 1:
#					self.push(a - 8)
#				else:
#					self.push(a - 1)
				self.push(a - 1)

			case Token(type=Operands.OP_MOD, value=val):
				b = self.pop()
				a = self.pop()
				self.push(a % b)

			case Token(type=Operands.OP_BLSH, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b << a)

			case Token(type=Operands.OP_BRSH, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b >> a)

			case Token(type=Operands.OP_BAND, value=val):
				a = self.pop()
				b = self.pop()
				if self.last_case == 0:
					self.push(b & a)
				if self.last_case == 1:
					self.push(b and a)

			case Token(type=Operands.OP_BOR, value=val):
				a = self.pop()
				b = self.pop()
				if self.last_case == 0:
					self.push(b | a)
				if self.last_case == 1:
					self.push(b or a)

			case Token(type=Operands.OP_BXOR, value=val):
				b = self.pop()
				a = self.pop()
				if self.last_case == 0:
					self.push(b ^ a)
				if self.last_case == 1:
					self.push(bool(b ^ a))

			case Token(type=Operands.OP_EQ, value=val):
				a = self.pop()
				b = self.pop()
				self.push(a == b)

			case Token(type=Operands.OP_NE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b != a)

			case Token(type=Operands.OP_GT, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b > a)

			case Token(type=Operands.OP_GE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b >= a)

			case Token(type=Operands.OP_LT, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b < a)

			case Token(type=Operands.OP_LE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b <= a)

			case Token(type=OpTypes.OP_DUMP, value=val):
				self.fds[1].write(str(self.pop()).encode('utf8'))
				self.fds[1].write(b'\n')

			case Token(type=OpTypes.OP_CDUMP, value=val):
				self.fds[1].write(chr(self.pop()).encode('utf8'))

			case Token(type=OpTypes.OP_UDUMP, value=val):
				self.fds[1].write(str(ctypes.c_ulonglong(self.pop()).value).encode('utf8'))
				self.fds[1].write(b'\n')

			case Token(type=OpTypes.OP_HEXDUMP, value=val):
				self.fds[1].write(hex(self.pop()).encode('utf8'))
				self.fds[1].write(b'\n')

			case Token(type=OpTypes.OP_SYSCALL, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_SYSCALL1, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_SYSCALL2, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_SYSCALL3, value=val):
				syscall = self.pop()
				arg1 = self.pop()
				arg2 = self.pop()
				arg3 = self.pop()
				if syscall == 0:
					b = self.fds[arg1].read(arg3)
					self.memory[arg2:arg2+len(b)] = b
				elif syscall == 1:
					m = bytes(takewhile(lambda x: x != 0, self.memory[arg2:arg2+arg3])).decode('utf8')
					
					self.fds[arg1].write(m.encode('utf8'))
					self.fds[arg1].flush()
					self.push(len(m))
				else:
					raise RuntimeError(NotImplemented, instruction, syscall, arg1, arg2, arg3)

			case Token(type=OpTypes.OP_SYSCALL4, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_SYSCALL5, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_SYSCALL6, value=val):
				raise RuntimeError(NotImplemented, instruction)


			case Token(type=OpTypes.OP_RSYSCALL1, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL2, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL3, value=val):
				arg3 = self.pop()
				arg2 = self.pop()
				arg1 = self.pop()
				syscall = self.pop()
				if syscall == 0:
					b = self.fds[arg1].read(arg3)
					self.memory[arg2:arg2+len(b)] = b
				elif syscall == 1:
					m = bytes(takewhile(lambda x: x != 0, self.memory[arg2:arg2+arg3])).decode('utf8')
					self.fds[arg1].write(m.encode('utf8'))
					self.fds[arg1].flush()
					self.push(len(m))
				else:
					raise RuntimeError(NotImplemented, instruction, syscall, arg1, arg2, arg3)

			case Token(type=OpTypes.OP_RSYSCALL4, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL5, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL6, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=FlowControl.OP_IF, value=val, position=p):
				pass

			case Token(type=FlowControl.OP_ELIF, value=val, position=p):
				pass

			case Token(type=FlowControl.OP_ELSE, value=val, position=p):
				if val.next:
					return val.next.position - p
				return val.end.position - p

			case Token(type=FlowControl.OP_WHILE, value=val):
				pass

			case Token(type=FlowControl.OP_DO, value=val, position=p):
				a = self.pop()
				if a == 0:
					if val.next:
						return val.next.position - p
					return val.end.position - p

			case Token(type=FlowControl.OP_END, value=val, position=p):
				if val.root.type in [FlowControl.OP_WHILE,]:
					return val.root.position - p

			case Token(type=Intrinsics.OP_ARGC, value=val):
				self.push(self.argc)

			case Token(type=Intrinsics.OP_ARGV, value=val):
				self.push(self.argv_ptr)

			case Token(type=OpTypes.OP_STORE, value=val):
				addr = self.pop()
				value = self.pop()
				self.memory[addr] = value & 0xFF

			case Token(type=OpTypes.OP_LOAD, value=val):
				addr = self.pop()
				self.push(self.memory[addr])

			case Token(type=OpTypes.OP_STORE16, value=val):
				addr = self.pop()
				value = (self.pop() & 0xFFFF).to_bytes(length=2, byteorder='little')
				self.memory[addr:addr+2] = value

			case Token(type=OpTypes.OP_LOAD16, value=val):
				addr = self.pop()
				self.push(int.from_bytes(self.memory[addr:addr+2], byteorder="little"))

			case Token(type=OpTypes.OP_STORE32, value=val):
				addr = self.pop()
				value = (self.pop() & 0xFFFFFFFF).to_bytes(length=4, byteorder='little')
				self.memory[addr:addr+4] = value

			case Token(type=OpTypes.OP_LOAD32, value=val):
				addr = self.pop()
				self.push(int.from_bytes(self.memory[addr:addr+4], byteorder="little"))

			case Token(type=OpTypes.OP_STORE64, value=val):
				addr = self.pop()
				value = (self.pop() & 0xFFFFFFFFFFFFFFFF).to_bytes(length=8, byteorder='little')
				self.memory[addr:addr+8] = value

			case Token(type=OpTypes.OP_LOAD64, value=val):
				addr = self.pop()
				self.push(int.from_bytes(self.memory[addr:addr+8], byteorder="little"))

			case Token(type=OpTypes.OP_EXIT, value=val):
				raise self.ExitFromEngine(self.pop())

			case Token(type=OpTypes.OP_WORD):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=FlowControl.OP_LABEL):
				pass

			case Token(type=PreprocTypes.CAST):
				pass

			case _:
				raise RuntimeError(NotImplemented, instruction)
		return 0

	def close(self, program):
		pass
