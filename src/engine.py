from abc import ABC, abstractmethod
from typing import Any, TextIO, Generator, Iterator
from functools import cached_property
from queue import LifoQueue, Empty

import ctypes
import sys

from .lexer import (
	Token,
	TokenInfo,
	TokenTypes,

	OpTypes,
	PreprocTypes,
	Intrinsics,
	Operands,
	FlowControl,

	LangExceptions,
	UnknownToken,
	InvalidSyntax,
	SymbolRedefined,
	FileError,
)

STR_CAPACITY = 640_000
MEM_CAPACITY = 640_000


class Engine(ABC):
	class ExitFromEngine(Exception): pass

	@abstractmethod
	def __init__(self, buffer: TextIO):
		...

	@abstractmethod
	def step(self, instruction: Token) -> int:
		...

	@abstractmethod
	def close(self):
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
		return f"{self.name}:"

	@cached_property
	def size(self) -> int:
		return -1



class Compiler(Engine):
	def __init__(self, buffer: TextIO):
		self.buffer: TextIO = buffer
		self._code = []
		self.strs = []
		self.labels = 0

		self.block("SLANG COMPILED PROGRAM", None)
		self.asm1('extern', '__dump')
		self.asm1('extern', '__udump')
		self.asm1('extern', '__hexdump')
		self.asm1('extern', '__cdump')
		self.asm1('segment', '.text')
		self.asm1('global', '_start')
		self.label('\n_start')

	def block(self, comment: str, token: Token | None) -> None:
		self._code.append([Block(comment)])
		if token and 1:
			self.label(f"{token.type.name}_{len(self._code)}_{len(self._code[-1])}", nl=True)

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
		if len(args) < 7:
			for reg, arg in reversed([*zip(registers, args)]):
				if arg is None:
					self.asm1("pop", f"{reg}")
				else:
					self.asm2("mov", f"{reg}", f"{arg}")

		self.asm1("push", "rbp")
		self.asm2("sub", "rsp", "8")
		self.asm2("mov", "rbp", "rsp")
		self.asm1("call", f"{name}")
		self.asm2("add", "rsp", "8")
		self.asm1("pop", "rbp")

	def step(self, instruction: Token) -> int:
		match instruction:
			case Token(type=FlowControl.OP_LABEL, value=val):
				self.block("label", instruction)
				self.label(val, force_unique=True)

			case Token(type=OpTypes.OP_PUSH, value=val):
				self.block("push", instruction)
				self.asm1("push", f"{val:.0f}")

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

			case Token(type=Operands.OP_PLUS, value=val):
				self.block("plus", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
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

			case Token(type=Operands.OP_BXOR, value=val):
				self.block("bitwise xor", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm2("xor", "rax", "rsi")
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
				self.asm1("pop", "rdi")
				self.asm2("mov", "rax", "60")
				self.asm("syscall")
				self.asm1("push", "rax")

				raise self.ExitFromEngine(0)

			case Token(type=FlowControl.OP_IF, value=val):
				self.block("IF", instruction)
				self.label(instruction.label())
				self.asm1("pop", "rax")
				self.asm2("test", "rax", "rax")
				self.asm1("jz", f"{val['token'].label()}")

			case Token(type=FlowControl.OP_ELSE, value=val):
				self.block("ELSE", instruction)
				self.asm1("jmp", f"{val['token'].label()}")
				self.label(instruction.label())


			case Token(type=FlowControl.OP_WHILE, value=val):
				self.block("WHILE", instruction)
				self.label(instruction.label())

			case Token(type=FlowControl.OP_DO, value=val):
				self.block("DO", instruction)
				self.label(instruction.label())
				self.asm1("pop", "rax")
				self.asm2("test", "rax", "rax")
				self.asm1("jz", f"{val['token'].label()}")

			case Token(type=FlowControl.OP_END, value=val):
				self.block("END", instruction)
				if val.type in [FlowControl.OP_WHILE,]:
					self.asm1("jmp", f"{val.label()}")
				self.label(instruction.label())

			case Token(type=Intrinsics.OP_MEM, value=val):
				self.block("mem", instruction)
				self.asm1("push", "mem")

			case Token(type=OpTypes.OP_STORE, value=val):
				self.block("store", instruction)
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm2("mov", "byte [rax]", "bl")

			case Token(type=OpTypes.OP_LOAD, value=val):
				self.block("load", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "bl", "byte [rax]")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_STORE16, value=val):
				self.block("store 16", instruction)
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm2("mov", "word [rax]", "bx")

			case Token(type=OpTypes.OP_LOAD16, value=val):
				self.block("load 16", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "bx", "word [rax]")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_STORE32, value=val):
				self.block("store 32", instruction)
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm2("mov", "dword [rax]", "ebx")

			case Token(type=OpTypes.OP_LOAD32, value=val):
				self.block("load 32", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "ebx", "dword [rax]")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_STORE64, value=val):
				self.block("store 64", instruction)
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm2("mov", "[rax]", "rbx")

			case Token(type=OpTypes.OP_LOAD64, value=val):
				self.block("load 64", instruction)
				self.asm1("pop", "rax")
				self.asm2("xor", "rbx", "rbx")
				self.asm2("mov", "rbx", "[rax]")
				self.asm1("push", "rbx")


			case Token(type=OpTypes.OP_WORD):
				raise RuntimeError(NotImplemented, instruction)

			case _:
				raise RuntimeError(NotImplemented, instruction)

		return 0

	def close(self):
		self.block("DATA", None)
		self.asm1("segment", ".data")
		for index, s in enumerate(self.strs):
			self.label(f"STR_{index}", nl=False)
			self.asm1("db", ','.join(map(hex, s.encode('utf8'))))
		
		self.block("MEMORY", None)
		self.asm1("segment", ".bss")
		self.label("mem", nl=False)
		self.asm1("resb", f"{MEM_CAPACITY}")

		align = max([j.size for i in self._code for j in i])
		for i in self._code:
			for j in i:
				self.buffer.write(j % align)
			self.buffer.write('\n')

class Interpreter(Engine):
	def __init__(self, buffer: TextIO):
		self.queue = LifoQueue(-1)
		self.memory = bytearray(1 + MEM_CAPACITY + STR_CAPACITY)
		self.ptr = 1
		self.buffer = buffer
		self.setargv(sys.modules['__main__'].argv)

	def setargv(self, av):
		for i in av[::-1]:
			self.memory[self.ptr:self.ptr+len(i)+1] = i.encode("utf8") + b'\0'
			self.push(self.ptr)
			self.ptr += len(i) + 1
		self.push(len(av))

	def push(self, v: Any): self.queue.put(v)
	def pop(self) -> Any: return self.queue.get_nowait()

	def step(self, instruction: Token):
		match instruction:
			case Token(type=OpTypes.OP_PUSH, value=val):
				self.queue.put(val)

			case Token(type=OpTypes.OP_STRING, value=val):
				self.push(len(val))
				self.push(self.ptr)
				self.memory[self.ptr:self.ptr+len(val)+1] = val.encode("utf8") + b'\0'
				self.ptr += len(val) + 1
				# push len
				# push ptr to start of memory

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

			case Token(type=Operands.OP_PLUS, value=val):
				self.push(self.pop() + self.pop())

			case Token(type=Operands.OP_MINUS, value=val):
				a = self.pop()
				b = self.pop()
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
				self.push(b & a)
			case Token(type=Operands.OP_BOR, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b | a)
			case Token(type=Operands.OP_BXOR, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b ^ a)

			case Token(type=Operands.OP_EQ, value=val):
				self.push(int(self.pop() == self.pop()))

			case Token(type=Operands.OP_NE, value=val):
				b = self.pop()
				a = self.pop()
				self.push(int(a != b))

			case Token(type=Operands.OP_GT, value=val):
				b = self.pop()
				a = self.pop()
				self.push(int(b > a))

			case Token(type=Operands.OP_GE, value=val):
				b = self.pop()
				a = self.pop()
				self.push(int(b >= a))

			case Token(type=Operands.OP_LT, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(b < a))

			case Token(type=Operands.OP_LE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(b <= a))

			case Token(type=OpTypes.OP_DUMP, value=val):
				self.buffer.write(str(self.pop()))
				self.buffer.write('\n')

			case Token(type=OpTypes.OP_CDUMP, value=val):
				self.buffer.write(chr(self.pop()))

			case Token(type=OpTypes.OP_UDUMP, value=val):
				self.buffer.write(str(ctypes.c_ulonglong(self.pop()).value))
				self.buffer.write('\n')

			case Token(type=OpTypes.OP_HEXDUMP, value=val):
				self.buffer.write(hex(self.pop()))
				self.buffer.write('\n')

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
				if syscall == 1:
					if arg1 == 1:
						m = self.memory[arg2:arg2+arg3].decode('utf8')
						self.buffer.write(m)
						self.push(len(m))
					elif arg1 == 2:
						m = self.memory[arg2:arg2+arg3].decode('utf8')
						sys.stderr.write(m)
						self.push(len(m))
					else:
						raise RuntimeError(NotImplemented, instruction, syscall, arg1, arg2, arg3)
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
				if syscall == 1:
					if arg1 == 1:
						m = self.memory[arg2:arg2+arg3].decode('utf8')
						self.buffer.write(m)
						self.push(m)
					elif arg1 == 2:
						m = self.memory[arg2:arg2+arg3].decode('utf8')
						sys.stderr.write(m)
						self.push(m)
					else:
						raise RuntimeError(NotImplemented, instruction, syscall, arg1, arg2, arg3)

			case Token(type=OpTypes.OP_RSYSCALL4, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL5, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL6, value=val):
				raise RuntimeError(NotImplemented, instruction)




			case Token(type=FlowControl.OP_IF, value=val, position=p):
				a = self.pop()
				if a == 0:
					return val['token'].position - p

			case Token(type=FlowControl.OP_ELSE, value=val, position=p):
				return val['token'].position - p

			case Token(type=FlowControl.OP_WHILE, value=val):
				pass

			case Token(type=FlowControl.OP_DO, value=val, position=p):
				a = self.pop()
				if a == 0:
					return val['token'].position - p

			case Token(type=FlowControl.OP_END, value=val, position=p):
				if val.type in [FlowControl.OP_WHILE,]:
					return val.position - p
#					return -(val[0] + val[2].value[0])

			case Token(type=Intrinsics.OP_MEM, value=val):
				self.push(STR_CAPACITY)

			case Token(type=OpTypes.OP_STORE, value=val):
				value = self.pop()
				addr = self.pop()
				self.memory[addr] = value & 0xFF

			case Token(type=OpTypes.OP_LOAD, value=val):
				addr = self.pop()
				self.push(self.memory[addr])

			case Token(type=OpTypes.OP_STORE16, value=val):
				value = (self.pop() & 0xFFFF).to_bytes(length=2, byteorder='little')
				addr = self.pop()
				self.memory[addr:addr+2] = value

			case Token(type=OpTypes.OP_LOAD16, value=val):
				addr = self.pop()
				self.push(int.from_bytes(self.memory[addr:addr+2], byteorder="little"))

			case Token(type=OpTypes.OP_STORE32, value=val):
				value = (self.pop() & 0xFFFFFFFF).to_bytes(length=4, byteorder='little')
				addr = self.pop()
				self.memory[addr:addr+4] = value

			case Token(type=OpTypes.OP_LOAD32, value=val):
				addr = self.pop()
				self.push(int.from_bytes(self.memory[addr:addr+4], byteorder="little"))

			case Token(type=OpTypes.OP_STORE64, value=val):
				value = (self.pop() & 0xFFFFFFFFFFFFFFFF).to_bytes(length=8, byteorder='little')
				addr = self.pop()
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

			case _:
				raise RuntimeError(NotImplemented, instruction)
		return 0

	def close(self):
		pass
