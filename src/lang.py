from queue import LifoQueue, Empty
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Any

from functools import cached_property
import subprocess
import sys
import os
from pathlib import Path
import shutil

import argparse
import uuid
import re

import ctypes

from .lexer import (
	tokenize,
	Token,
	TokenInfo,
	TokenTypes,
	OpTypes,

	LangExceptions,
	UnknownToken,
	InvalidSyntax,
	SymbolRedefined,
)

STR_CAPACITY = 640_000
MEM_CAPACITY = 640_000

def PUSH(val: Any, info=None)    -> Token: return Token(OpTypes.OP_PUSH, val, info=info)
def STRING(val: str, info=None)  -> Token: return Token(OpTypes.OP_STRING, val, info=info)
def WORD(val: str, info=None)    -> Token: return Token(OpTypes.OP_WORD, val, info=info)
def DROP(info=None)              -> Token: return Token(OpTypes.OP_DROP, info=info)
def DUP(info=None)               -> Token: return Token(OpTypes.OP_DUP, info=info)
def DUP2(info=None)              -> Token: return Token(OpTypes.OP_DUP2, info=info)
def SWAP(info=None)              -> Token: return Token(OpTypes.OP_SWAP, info=info)
def OVER(info=None)              -> Token: return Token(OpTypes.OP_OVER, info=info)

def PLUS(info=None)              -> Token: return Token(OpTypes.OP_PLUS, info=info)
def MINUS(info=None)             -> Token: return Token(OpTypes.OP_MINUS, info=info)
def MUL(info=None)               -> Token: return Token(OpTypes.OP_MUL, info=info)
def DIV(info=None)               -> Token: return Token(OpTypes.OP_DIV, info=info)
def MOD(info=None)               -> Token: return Token(OpTypes.OP_MOD, info=info)
def DIVMOD(info=None)            -> Token: return Token(OpTypes.OP_DIVMOD, info=info)

def OP_BLSH(info=None)           -> Token: return Token(OpTypes.OP_BLSH, info=info)
def OP_BRSH(info=None)           -> Token: return Token(OpTypes.OP_BRSH, info=info)
def OP_BAND(info=None)           -> Token: return Token(OpTypes.OP_BAND, info=info)
def OP_BOR(info=None)            -> Token: return Token(OpTypes.OP_BOR, info=info)
def OP_BXOR(info=None)           -> Token: return Token(OpTypes.OP_BXOR, info=info)

def EQ(info=None)                -> Token: return Token(OpTypes.OP_EQ, info=info)
def NE(info=None)                -> Token: return Token(OpTypes.OP_NE, info=info)
def GT(info=None)                -> Token: return Token(OpTypes.OP_GT, info=info)
def GE(info=None)                -> Token: return Token(OpTypes.OP_GE, info=info)
def LT(info=None)                -> Token: return Token(OpTypes.OP_LT, info=info)
def LE(info=None)                -> Token: return Token(OpTypes.OP_LE, info=info)

def DUMP(info=None)              -> Token: return Token(OpTypes.OP_DUMP, info=info)
def UDUMP(info=None)             -> Token: return Token(OpTypes.OP_UDUMP, info=info)
def CDUMP(info=None)             -> Token: return Token(OpTypes.OP_CDUMP, info=info)
def HEXDUMP(info=None)           -> Token: return Token(OpTypes.OP_HEXDUMP, info=info)
def PRINTLINE(info=None)         -> Token: return Token(OpTypes.OP_PRINTLINE, info=info)

def SYSCALL(info=None)           -> Token: return Token(OpTypes.OP_SYSCALL, info=info)
def SYSCALL1(info=None)          -> Token: return Token(OpTypes.OP_SYSCALL1, info=info)
def SYSCALL2(info=None)          -> Token: return Token(OpTypes.OP_SYSCALL2, info=info)
def SYSCALL3(info=None)          -> Token: return Token(OpTypes.OP_SYSCALL3, info=info)
def SYSCALL4(info=None)          -> Token: return Token(OpTypes.OP_SYSCALL4, info=info)
def SYSCALL5(info=None)          -> Token: return Token(OpTypes.OP_SYSCALL5, info=info)
def SYSCALL6(info=None)          -> Token: return Token(OpTypes.OP_SYSCALL6, info=info)

def RSYSCALL1(info=None)         -> Token: return Token(OpTypes.OP_RSYSCALL1, info=info)
def RSYSCALL2(info=None)         -> Token: return Token(OpTypes.OP_RSYSCALL2, info=info)
def RSYSCALL3(info=None)         -> Token: return Token(OpTypes.OP_RSYSCALL3, info=info)
def RSYSCALL4(info=None)         -> Token: return Token(OpTypes.OP_RSYSCALL4, info=info)
def RSYSCALL5(info=None)         -> Token: return Token(OpTypes.OP_RSYSCALL5, info=info)
def RSYSCALL6(info=None)         -> Token: return Token(OpTypes.OP_RSYSCALL6, info=info)

def IF(info=None)                -> Token: return Token(OpTypes.OP_IF, info=info)
def ELSE(info=None)              -> Token: return Token(OpTypes.OP_ELSE, info=info)
def WHILE(info=None)             -> Token: return Token(OpTypes.OP_WHILE, info=info)
def DO(info=None)                -> Token: return Token(OpTypes.OP_DO, info=info)
def MACRO(info=None)             -> Token: return Token(OpTypes.OP_MACRO, info=info)
def END(info=None)               -> Token: return Token(OpTypes.OP_END, info=info)
def LABEL(name, info=None)       -> Token: return Token(OpTypes.OP_LABEL, name, info=info)

def MEM(info=None)               -> Token: return Token(OpTypes.OP_MEM, info=info)
def STORE(info=None)             -> Token: return Token(OpTypes.OP_STORE, info=info)
def LOAD(info=None)              -> Token: return Token(OpTypes.OP_LOAD, info=info)

def EXIT(code: int=0, info=None) -> Token: return Token(OpTypes.OP_EXIT, value=code, info=info)


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

class Compiler:
	def __init__(self, buffer):
		self.buffer = buffer
		self._code = []
		self.strs = []
		self.labels = 0
		self.block("SLANG COMPILED PROGRAM", None)
		self.asm1('extern', '__dump')
		self.asm1('extern', '__udump')
		self.asm1('extern', '__hexdump')
		self.asm1('extern', '__cdump')
		self.asm1('extern', '__printline')
		self.asm1('segment', '.text')
		self.asm1('global', '_start')
		self.label('\n_start')

	def block(self, comment, token: Token | None):
		self._code.append([Block(comment)])
		if token and 0:
			self.label(f"{token.type.name}_{len(self._code)}_{len(self._code[-1])}", nl=True)

	def label(self, name, nl=True, force_unique=False):
		if force_unique:
			self._code[-1].append(Label(f"{name}_{self.labels}", nl=nl))
		else:
			self._code[-1].append(Label(name, nl=nl))
		self.labels += 1

	def asm(self, ins):
		self._code[-1].append(ASM(ins))

	def asm1(self, ins, arg):
		self._code[-1].append(ASM1(ins, arg))

	def asm2(self, ins, arg1, arg2):
		self._code[-1].append(ASM2(ins, arg1, arg2))

	def call_cfunction(self, name: str, args: list[Any | None]):
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

	def step(self, instruction: Token):
		assert OpTypes.OP_COUNT.value == 54, f"Not all operators are handled {OpTypes.OP_COUNT.value}"
		match instruction:
			case Token(type=OpTypes.OP_LABEL, value=val):
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

			case Token(type=OpTypes.OP_DROP, value=val):
				self.block("pop", instruction)
				self.asm1("pop", "rax")

			case Token(type=OpTypes.OP_DUP, value=val):
				self.block("dup", instruction)
				self.asm1("pop", "rax")
				self.asm1("push", "rax")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_DUP2, value=val):
				self.block("dup", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_SWAP, value=val):
				self.block("swap", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm1("push", "rax")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_OVER, value=val):
				self.block("over", instruction)
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm1("push", "rax")
				self.asm1("push", "rbx")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_PLUS, value=val):
				self.block("plus", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("add", "rax", "rbx")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_MUL, value=val):
				self.block("mul", instruction)
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm1("mul", "rcx")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_DIV, value=val):
				self.block("div", instruction)
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_MOD, value=val):
				self.block("div", instruction)
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rdx")

			case Token(type=OpTypes.OP_DIVMOD, value=val):
				self.block("divmod", instruction)
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rax")
				self.asm1("push", "rdx")

			case Token(type=OpTypes.OP_BLSH, value=val):
				self.block("bitwise shift left", instruction)
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm2("shl", "rax", "cl")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_BRSH, value=val):
				self.block("bitwise shift right", instruction)
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm2("shr", "rax", "cl")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_BAND, value=val):
				self.block("bitwise and", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm2("and", "rax", "rsi")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_BOR, value=val):
				self.block("bitwise or", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm2("or", "rax", "rsi")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_BXOR, value=val):
				self.block("bitwise xor", instruction)
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm2("xor", "rax", "rsi")
				self.asm1("push", "rax")


			case Token(type=OpTypes.OP_MINUS, value=val):
				self.block("minus", instruction)
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("sub", "rbx", "rax")
				self.asm1("push", "rbx")

			case Token(type=OpTypes.OP_EQ, value=val):
				self.block("eq", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmove", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=OpTypes.OP_NE, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovne", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=OpTypes.OP_GT, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovg", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=OpTypes.OP_GE, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovge", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=OpTypes.OP_LT, value=val):
				self.block("ne", instruction)
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovl", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=OpTypes.OP_LE, value=val):
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

			case Token(type=OpTypes.OP_PRINTLINE, value=val):
				self.block("printline", instruction)
				self.call_cfunction("__printline", [None])

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
				self.asm2("mov", "rax", "60")
				self.asm2("mov", "rdi", f"{val}")
				self.asm("syscall")
				self.asm1("push", "rax")

			case Token(type=OpTypes.OP_IF, value=val):
				self.block("IF", instruction)
				self.label(instruction.label())
				self.asm1("pop", "rax")
				self.asm2("test", "rax", "rax")
				self.asm1("jz", f"{val['token'].label()}")

			case Token(type=OpTypes.OP_ELSE, value=val):
				self.block("ELSE", instruction)
				self.asm1("jmp", f"{val['token'].label()}")
				self.label(instruction.label())


			case Token(type=OpTypes.OP_WHILE, value=val):
				self.block("WHILE", instruction)
				self.label(instruction.label())

			case Token(type=OpTypes.OP_DO, value=val):
				self.block("DO", instruction)
				self.label(instruction.label())
				self.asm1("pop", "rax")
				self.asm2("test", "rax", "rax")
				self.asm1("jz", f"{val['token'].label()}")

			case Token(type=OpTypes.OP_END, value=val):
				self.block("END", instruction)
				if val.type in [OpTypes.OP_WHILE,]:
					self.asm1("jmp", f"{val.label()}")
				self.label(instruction.label())

			case Token(type=OpTypes.OP_MEM, value=val):
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

class Interpreter:
	def __init__(self):
		self.queue = LifoQueue(-1)
		self.memory = bytearray(MEM_CAPACITY + STR_CAPACITY)
		self.ptr = 0

	def close(self):
		pass
	def push(self, v: Any): self.queue.put(v)
	def pop(self) -> Any: return self.queue.get_nowait()
	def step(self, instruction: Token):
		assert OpTypes.OP_COUNT.value == 54, f"Not all operators are handled {OpTypes.OP_COUNT.value}"

		print(instruction)
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

			case Token(type=OpTypes.OP_DROP, value=val):
				self.pop()

			case Token(type=OpTypes.OP_DUP, value=val):
				a = self.pop()
				self.push(a)
				self.push(a)

			case Token(type=OpTypes.OP_DUP2, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b)
				self.push(a)
				self.push(b)
				self.push(a)

			case Token(type=OpTypes.OP_SWAP, value=val):
				a = self.pop()
				b = self.pop()
				self.push(a)
				self.push(b)

			case Token(type=OpTypes.OP_OVER, value=val):
				b = self.pop()
				a = self.pop()
				self.push(a)
				self.push(b)
				self.push(a)

			case Token(type=OpTypes.OP_PLUS, value=val):
				self.push(self.pop() + self.pop())

			case Token(type=OpTypes.OP_MINUS, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b - a)

			case Token(type=OpTypes.OP_MUL, value=val):
				self.push(self.pop() * self.pop())

			case Token(type=OpTypes.OP_DIV, value=val):
				self.push(self.pop() // self.pop())

			case Token(type=OpTypes.OP_MOD, value=val):
				a = self.pop()
				b = self.pop()
				self.push(a // b)
				self.push(a % b)

			case Token(type=OpTypes.OP_MOD, value=val):
				self.push(self.pop() % self.pop())

			case Token(type=OpTypes.OP_BLSH, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b << a)
			case Token(type=OpTypes.OP_BRSH, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b >> a)
			case Token(type=OpTypes.OP_BAND, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b & a)
			case Token(type=OpTypes.OP_BOR, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b | a)
			case Token(type=OpTypes.OP_BXOR, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b ^ a)

			case Token(type=OpTypes.OP_EQ, value=val):
				self.push(int(self.pop() == self.pop()))

			case Token(type=OpTypes.OP_NE, value=val):
				self.push(int(self.pop() != self.pop()))

			case Token(type=OpTypes.OP_GT, value=val):
				b = self.pop()
				a = self.pop()
				self.push(int(b > a))

			case Token(type=OpTypes.OP_GE, value=val):
				b = self.pop()
				a = self.pop()
				self.push(int(b >= a))

			case Token(type=OpTypes.OP_LT, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(b < a))

			case Token(type=OpTypes.OP_LE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(b <= a))

			case Token(type=OpTypes.OP_DUMP, value=val):
				print(self.pop())

			case Token(type=OpTypes.OP_CDUMP, value=val):
				print(chr(self.pop()), end='')

			case Token(type=OpTypes.OP_UDUMP, value=val):
				print(ctypes.c_ulonglong(self.pop()).value)

			case Token(type=OpTypes.OP_HEXDUMP, value=val):
				print(hex(self.pop()))

			case Token(type=OpTypes.OP_PRINTLINE, value=val):
				addr = self.pop()
				for i in self.memory[addr:]:
					if not i: break
					print(chr(i), end='')

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
						print(m, end='')
						self.push(len(m))
					elif arg1 == 2:
						m = self.memory[arg2:arg2+arg3].decode('utf8')
						print(m, end='', file=sys.stderr)
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
						print(m, end='')
						self.push(m)
					elif arg1 == 2:
						m = self.memory[arg2:arg2+arg3].decode('utf8')
						print(m, end='', file=sys.stderr)
						self.push(m)
					else:
						raise RuntimeError(NotImplemented, instruction, syscall, arg1, arg2, arg3)

			case Token(type=OpTypes.OP_RSYSCALL4, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL5, value=val):
				raise RuntimeError(NotImplemented, instruction)

			case Token(type=OpTypes.OP_RSYSCALL6, value=val):
				raise RuntimeError(NotImplemented, instruction)




			case Token(type=OpTypes.OP_IF, value=val):
				a = self.pop()
				if a == 0:
					return val['token']

			case Token(type=OpTypes.OP_ELSE, value=val, position=p):
				return val['token'].position - p

			case Token(type=OpTypes.OP_WHILE, value=val):
				pass

			case Token(type=OpTypes.OP_DO, value=val, position=p):
				a = self.pop()
				if a == 0:
					return val['token'].position - p

			case Token(type=OpTypes.OP_END, value=val, position=p):
				if val.type in [OpTypes.OP_WHILE,]:
					return val.position - p
#					return -(val[0] + val[2].value[0])

			case Token(type=OpTypes.OP_MEM, value=val):
				self.push(STR_CAPACITY)

			case Token(type=OpTypes.OP_STORE, value=val):
				value = self.pop()
				addr = self.pop()
				self.memory[addr] = value & 0xFF

			case Token(type=OpTypes.OP_LOAD, value=val):
				addr = self.pop()
				self.push(self.memory[addr])

			case Token(type=OpTypes.OP_EXIT, value=val):
				exit(val)

			case Token(type=OpTypes.OP_WORD):
				raise RuntimeError(NotImplemented, instruction)

			case _:
				raise RuntimeError(NotImplemented, instruction)
		#print(f"{str(instruction)} {self.queue.queue} {self.memory[:50]}", file=sys.stderr)
		return 0

def unescape_string(s):
	return s.encode('latin-1', 'backslashreplace').decode('unicode-escape')

BUILTIN_WORDS = [
	"dump",
	"udump",
	"blsh",
	"brsh",
	"band",
	"bor",
	"bxor",
	"cdump",
	"hexdump",
	"printline",
	"syscall",
	"syscall1",
	"syscall2",
	"syscall3",
	"syscall4",
	"syscall5",
	"syscall6",
	"rsyscall1",
	"rsyscall2",
	"rsyscall3",
	"rsyscall4",
	"rsyscall5",
	"rsyscall6",
	"drop",
	"dup",
	"dup2",
	"swap",
	"over",
	"exit",
	"if",
	"else",
	"while",
	"do",
	"macro",
	"end",
	"mem",
	"store",
	"load",
]





class Program:
	class Comment(Exception): ...
	class EndLine(Exception): ...
	def __init__(self, engine=None):
		self.instructions: list[Token] = []
		self.engine = engine
		self.pointer = 0
		self.symbols = {}
		self._in_macro = 0
		self._position = 0

	@classmethod
	def fromfile(cls, path):
		with open(path, 'r') as f:
			return cls.frombuffer(f)

	def match_token(self, token) -> list[Token]:
		assert OpTypes.OP_COUNT.value == 54, f"Not all operators are handled {OpTypes.OP_COUNT.value}"
		match token:
			case TokenInfo(type=TokenTypes.NUMBER, string=s): return [PUSH(int(s), info=token)]
			case TokenInfo(type=TokenTypes.CHAR, string=s): return [PUSH(ord(unescape_string(s)), info=token)]
			case TokenInfo(type=TokenTypes.STRING, string=s): return [STRING(unescape_string(s), info=token)]
			case TokenInfo(type=TokenTypes.OP, string="+"): return [PLUS(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="-"): return [MINUS(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="*"): return [MUL(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="/"): return [DIV(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="%"): return [MOD(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="/%"): return [DIVMOD(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="=="): return [EQ(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="!="): return [NE(info=token)]
			case TokenInfo(type=TokenTypes.OP, string=">"): return [GT(info=token)]
			case TokenInfo(type=TokenTypes.OP, string=">="): return [GE(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="<"): return [LT(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="<="): return [LE(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="."): return [STORE(info=token)]
			case TokenInfo(type=TokenTypes.OP, string=","): return [LOAD(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="<<"): return [OP_BLSH(info=token)]
			case TokenInfo(type=TokenTypes.OP, string=">>"): return [OP_BRSH(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="&"): return [OP_BAND(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="|"): return [OP_BOR(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="^"): return [OP_BXOR(info=token)]
			case TokenInfo(type=TokenTypes.OP, string="^"): return [OP_BXOR(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="dump"): return [DUMP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="udump"): return [UDUMP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="blsh"): return [OP_BLSH(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="brsh"): return [OP_BRSH(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="band"): return [OP_BAND(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="bor"): return [OP_BOR(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="bxor"): return [OP_BXOR(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="cdump"): return [CDUMP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="hexdump"): return [HEXDUMP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="printline"): return [PRINTLINE(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall"): return [SYSCALL(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall1"): return [SYSCALL1(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall2"): return [SYSCALL2(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall3"): return [SYSCALL3(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall4"): return [SYSCALL4(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall5"): return [SYSCALL5(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="syscall6"): return [SYSCALL6(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="rsyscall1"): return [RSYSCALL1(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="rsyscall2"): return [RSYSCALL2(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="rsyscall3"): return [RSYSCALL3(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="rsyscall4"): return [RSYSCALL4(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="rsyscall5"): return [RSYSCALL5(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="rsyscall6"): return [RSYSCALL6(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="drop"): return [DROP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="dup"): return [DUP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="dup2"): return [DUP2(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="swap"): return [SWAP(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="over"): return [OVER(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="exit"): return [EXIT(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="if"): return [IF(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="else"): return [ELSE(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="while"): return [WHILE(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="do"): return [DO(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="macro"): return [MACRO(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="end"): return [END(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="mem"): return [MEM(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="store"): return [STORE(info=token)]
			case TokenInfo(type=TokenTypes.WORD, string="load"): return [LOAD(info=token)]

			case TokenInfo(type=TokenTypes.WORD, string=s):
				if self._in_macro == 1:
					return [WORD(val=s, info=token)]
				if s not in self.symbols:
					raise UnknownToken(token, "Is not a registered or builtin symbol")
				return [LABEL(name=s, info=token), *[i.copy() for i in self.symbols[s].value]]

			case TokenInfo(type=TokenTypes.OP, string="//"): raise self.Comment()
			case TokenInfo(type=TokenTypes.NEW_LINE): raise self.EndLine()

			case _:
				raise UnknownToken(token, "Is not a recognized symbol")

	@classmethod
	def frombuffer(cls, buffer, *, debug=False):
		tokenizer = tokenize(buffer, debug=debug)

		self = cls()
		self.parse_tokens(tokenizer, debug=debug)
		return self

	
	def parse_tokens(self, tokenizer, *, debug=False):
		def run():
			comment = -1
			for token in tokenizer:
				if token.start[0] == comment: continue
				try:
					tokens = self.match_token(token)
					if self._in_macro:
						self._in_macro += 1
					for index, t in enumerate(tokens):
						t.position = self._position + index
						if debug:
							print(t)
						self.add(t)
					for t in tokens:
						yield t
				except self.Comment as e: comment = token.start[0]
				except self.EndLine as e: pass
		flow_control = self.process_flow_control()
		next(flow_control)
		for t in run():
			flow_control.send(t)
		flow_control.send(None)

#		for token in self.instructions:
#			print(token, token.value)
		return self

	def parse_macro(self, tokens):
		if len(tokens) == 1:
			raise InvalidSyntax(tokens[0].info, "`macro` requires a name")
		if tokens[1].type != OpTypes.OP_WORD:
			raise InvalidSyntax(tokens[1].info, f"`macro` name must be a word not `{tokens[1].type.name}`")
		if tokens[1].info.string in BUILTIN_WORDS:
			raise SymbolRedefined(tokens[1].info, "Is a builtin symbol")
		if tokens[1].info.string in self.symbols:
			raise SymbolRedefined(tokens[1].info, "Has already been defined")
			
		self.symbols[tokens[1].info.string] = tokens[0]
		tokens[0].value = tokens[2:]


	def process_flow_control(self):
		assert OpTypes.OP_COUNT.value == 54, f"Not all operators are handled {OpTypes.OP_COUNT.value}"
		stack = []

		while True:
			token = (yield)
			if token is None:
				break
			match token:
				case Token(type=OpTypes.OP_IF):
					token.value = {"token": token, "previous": token}
					stack.append(token.value)

				case Token(type=OpTypes.OP_ELSE):
					try:
						d = stack.pop()
					except IndexError:
						raise InvalidSyntax(token.info, "`else` requires a preceding `if`")
					token.value = {"token": token, "previous": d['previous']}
					stack.append(token.value)

					d['token'].value['token'] = token

				case Token(type=OpTypes.OP_END):
					try:
						d = stack.pop()
					except IndexError:
						raise InvalidSyntax(token.info, "`end` requires a preceding `if`, `else`, `do` or `macro`")

					d['token'].value['token'] = token
					token.value = d['previous']

					if token.value.type == OpTypes.OP_MACRO:
						self.parse_macro(self.instructions[token.value.position:token.position])
						for i in reversed(range(token.value.position, token.position+1)):
							self.instructions.pop(i)
							self._position -= 1
						self._in_macro = 0

				case Token(type=OpTypes.OP_WHILE):
					token.value = {"token": token, "previous": token}
					stack.append(token.value)

				case Token(type=OpTypes.OP_DO):
					try:
						d = stack.pop()
					except IndexError:
						raise InvalidSyntax(token.info, "`do` requires a preceding `while`")
					token.value = {"token": token, "previous": d['previous']}
					stack.append(token.value)

				case Token(type=OpTypes.OP_MACRO):
					token.value = {"token": token, "previous": token}
					stack.append(token.value)
					if self._in_macro:
						raise InvalidSyntax(token.info, f"nested `macro` definition is not allowed")

					self._in_macro = 1

		if stack:
			raise InvalidSyntax(stack[-1]['token'].info, "is missing end")
		yield

	def add(self, token: Token) -> 'Program':
		self.instructions.append(token)
		self._position += 1
		return self

	def run(self):
		if self.engine is None:
			raise ValueError("Add engine before running")
		skip = 0
		while self.pointer < len(self.instructions):
			self.pointer += self.engine.step(self.instructions[self.pointer]) + 1
		self.engine.close()

#		for i in self.instructions:
#			if skip:
#				skip -= 1
#				continue
#			skip += self.engine.step(i)

def callcmd(cmd, verbose=False):
	if verbose:
		print("CMD:", cmd)
		return subprocess.call(cmd)
	else:
		return subprocess.call(cmd, stdout=subprocess.DEVNULL)



def main(ac: int, av: list[str]):
	parser = argparse.ArgumentParser(prog="slang", description="A stack based language written in python")
	parser.add_argument("engine", choices=["interpret", "compile", "fclean", "lex"])
	parser.add_argument("-s", "--source")
	parser.add_argument("-o", "--output", default="a.out")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-e", "--exec", action="store_true")
	args = parser.parse_args()

	if args.source:
		with open(args.source, 'r') as f:
			try:
				p = Program.frombuffer(f, debug=args.engine=='lex')
			except LangExceptions as e:
				token: TokenInfo = e.args[0]
				msg = e.args[1]
				print(f"\033[31mError: line {token.start[0] + 1}: {e.__class__.__name__}:\033[0m\n")
				print(token.error())
				print(msg)
#				print(token)
				exit()
	else:
		p = (Program()
			.add(PUSH(35))
			.add(PUSH(35))
			.add(PLUS())
			.add(PUSH(1))
			.add(MINUS())
			.add(DUMP())
			.add(EXIT())
		)
	if args.engine == 'fclean':
		objs = Path("objs")
		if args.verbose:
			print("rm -rf objs")
		try:
			shutil.rmtree(objs)
		except FileNotFoundError:
			pass
		callcmd(["make", "-C", "src/cfunc/", "fclean"], verbose=args.verbose)

	if args.engine == 'interpret':
		p.engine = Interpreter()
		p.run()
	if args.engine == 'compile':
		objs = Path("objs")
		if not objs.exists():
			objs.mkdir()
		with open(objs / "intermediary.asm", 'w') as f:
			p.engine = Compiler(f)
			p.run()

		if e:=callcmd(["make", "-C", "src/cfunc/"], verbose=args.verbose):
			exit(e)
		if e:=callcmd(["nasm", "-f", "elf64", "objs/intermediary.asm", "-o", "objs/intermediary.o"], verbose=args.verbose):
			exit(e)
		if e:=callcmd(["ld", "src/cfunc/objs/dump.o", "objs/intermediary.o", "-lc", "-I", "/lib64/ld-linux-x86-64.so.2", "-o", args.output ], verbose=args.verbose):
			exit(e)

		if args.exec:
			if e:=callcmd([f"./{args.output}"], verbose=True):
				exit(e)
