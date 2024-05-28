from queue import LifoQueue, Empty
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Optional, Any

from functools import cached_property
import subprocess
import sys
import os
from pathlib import Path
import shutil

import argparse
import tokenize
import uuid


def iota(reset=False, *, v=[-1]):
	if reset:
		v[0] = -1
	v[0] += 1
	return v[0]

class UnknownToken(Exception):
	pass
class InvalidSyntax(Exception):
	pass

MEM_CAPACITY = 640_000

class TokenTypes(Enum):
	OP_PUSH		= iota(True)
	OP_POP		= iota()
	OP_DUP		= iota()

	OP_PLUS		= iota()
	OP_MINUS	= iota()
	OP_MUL		= iota()
	OP_DIV		= iota()

	OP_EQ		= iota()
	OP_NE		= iota()
	OP_GT		= iota()
	OP_GE		= iota()
	OP_LT		= iota()
	OP_LE		= iota()

	OP_DUMP		= iota()
	OP_UDUMP	= iota()
	OP_HEXDUMP	= iota()
	OP_IF		= iota()
	OP_ELSE		= iota()
	OP_WHILE	= iota()
	OP_DO		= iota()
	OP_END		= iota()

	OP_MEM		= iota()
	OP_STORE	= iota()
	OP_LOAD		= iota()

	OP_EXIT		= iota()
	OP_COUNT	= iota()

class Token:
	__slots__ = ("type", "value", "info", "id")

	type: TokenTypes
	value: Any
	info: tokenize.TokenInfo | None
	id: str

	def __init__(self, type: TokenTypes, value: Any=None, info=None):
		self.value = value
		self.type = type
		self.info = info
		self.id = str(uuid.uuid4())[:8]

	def __repr__(self):
		return f"{self.type}{f" {self.value}" if self.value else ""}"

	def label(self):
		return f"{self.type.name}_{self.id}"

def PUSH(val: Any, info=None) -> Token: return Token(TokenTypes.OP_PUSH, val, info=info)
def POP(info=None) -> Token: return Token(TokenTypes.OP_POP, info=info)
def DUP(info=None) -> Token: return Token(TokenTypes.OP_DUP, info=info)
def PLUS(info=None) -> Token: return Token(TokenTypes.OP_PLUS, info=info)
def MINUS(info=None) -> Token: return Token(TokenTypes.OP_MINUS, info=info)
def MUL(info=None) -> Token: return Token(TokenTypes.OP_MUL, info=info)
def DIV(info=None) -> Token: return Token(TokenTypes.OP_DIV, info=info)
def EQ(info=None) -> Token: return Token(TokenTypes.OP_EQ, info=info)
def NE(info=None) -> Token: return Token(TokenTypes.OP_NE, info=info)
def GT(info=None) -> Token: return Token(TokenTypes.OP_GT, info=info)
def GE(info=None) -> Token: return Token(TokenTypes.OP_GE, info=info)
def LT(info=None) -> Token: return Token(TokenTypes.OP_LT, info=info)
def LE(info=None) -> Token: return Token(TokenTypes.OP_LE, info=info)
def DUMP(info=None) -> Token: return Token(TokenTypes.OP_DUMP, info=info)
def UDUMP(info=None) -> Token: return Token(TokenTypes.OP_UDUMP, info=info)
def HEXDUMP(info=None) -> Token: return Token(TokenTypes.OP_HEXDUMP, info=info)
def IF(info=None) -> Token: return Token(TokenTypes.OP_IF, info=info)
def ELSE(info=None) -> Token: return Token(TokenTypes.OP_ELSE, info=info)
def WHILE(info=None) -> Token: return Token(TokenTypes.OP_WHILE, info=info)
def DO(info=None) -> Token: return Token(TokenTypes.OP_DO, info=info)
def END(info=None) -> Token: return Token(TokenTypes.OP_END, info=info)
def MEM(info=None) -> Token: return Token(TokenTypes.OP_MEM, info=info)
def STORE(info=None) -> Token: return Token(TokenTypes.OP_STORE, info=info)
def LOAD(info=None) -> Token: return Token(TokenTypes.OP_LOAD, info=info)
def EXIT(code: int=0, info=None) -> Token: return Token(TokenTypes.OP_EXIT, value=code, info=info)


class Instruction(ABC):
	@abstractmethod
	def align(self, largest) -> str:
		...

	@cached_property
	def size(self) -> int:
		...

	def __mod__(self, x: int):
		return f"{self.align(x)}\n"

class Comment(Instruction):
	def __init__(self, string):
		self.string = string

	def align(self, largest: int) -> str:
		return f"\t; {self.string}"

	@cached_property
	def size(self) -> int:
		return -1

class ASM(Instruction):
	def __init__(self, ins):
		self.ins = ins

	def align(self, largest: int) -> str:
		return f"\t{self.ins}"

	@cached_property
	def size(self) -> int:
		return len(self.ins) + 1

class ASM1(Instruction):
	def __init__(self, ins, arg0):
		self.ins = ins
		self.arg0 = arg0

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

	def align(self, largest: int) -> str:
		return f"\t{self.ins: <{largest}}{self.arg0},{self.arg1}"

	@cached_property
	def size(self) -> int:
		return len(self.ins) + 1

class Label(Instruction):
	def __init__(self, name):
		self.name = name

	def align(self, largest: int) -> str:
		return f"{self.name}:"

	@cached_property
	def size(self) -> int:
		return -1

class Compiler:
	def __init__(self, buffer):
		self.buffer = buffer
		self._code = []
		self.block("SLANG COMPILED PROGRAM")
		self.asm1('extern', '__dump')
		self.asm1('extern', '__udump')
		self.asm1('extern', '__hexdump')
		self.asm1('segment', '.text')
		self.asm1('global', '_start')
		self.label('\n_start')

	def block(self, comment):
		self._code.append([Comment(comment)])

	def label(self, name):
		self._code[-1].append(Label(name))

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

		self.block(f"{name} {args}")
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
		assert TokenTypes.OP_COUNT.value == 25, f"Not all operators are handled {TokenTypes.OP_COUNT.value}"
		match instruction:
			case Token(type=TokenTypes.OP_PUSH, value=val):
				self.block("push")
				self.asm1("push", f"{val:.0f}")

			case Token(type=TokenTypes.OP_POP, value=val):
				self.block("pop")
				raise RuntimeError("Not yet implemented")

			case Token(type=TokenTypes.OP_DUP, value=val):
				self.block("dup")
				self.asm1("pop", "rax")
				self.asm1("push", "rax")
				self.asm1("push", "rax")

			case Token(type=TokenTypes.OP_PLUS, value=val):
				self.block("plus")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("add", "rax", "rbx")
				self.asm1("push", "rax")

			case Token(type=TokenTypes.OP_MUL, value=val):
				self.block("mul")
				self.asm1("pop", "rcx")
				self.asm1("pop", "rax")
				self.asm1("mul", "rcx")
				self.asm1("push", "rax")

			case Token(type=TokenTypes.OP_DIV, value=val):
				self.block("div")
				self.asm2("xor", "edx", "edx")
				self.asm1("pop", "rsi")
				self.asm1("pop", "rax")
				self.asm1("div", "rsi")
				self.asm1("push", "rax")

			case Token(type=TokenTypes.OP_MINUS, value=val):
				self.block("minus")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("sub", "rbx", "rax")
				self.asm1("push", "rbx")

			case Token(type=TokenTypes.OP_EQ, value=val):
				self.block("eq")
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmove", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=TokenTypes.OP_NE, value=val):
				self.block("ne")
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovne", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=TokenTypes.OP_GT, value=val):
				self.block("ne")
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovg", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=TokenTypes.OP_GE, value=val):
				self.block("ne")
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovge", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=TokenTypes.OP_LT, value=val):
				self.block("ne")
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovl", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=TokenTypes.OP_LE, value=val):
				self.block("ne")
				self.asm2("xor", "rcx", "rcx")
				self.asm2("mov", "rdx", "1")
				self.asm1("pop", "rax")
				self.asm1("pop", "rbx")
				self.asm2("cmp", "rbx", "rax")
				self.asm2("cmovle", "rcx", "rdx")
				self.asm1("push", "rcx")

			case Token(type=TokenTypes.OP_DUMP, value=val):
				self.call_cfunction("__dump", [None])

			case Token(type=TokenTypes.OP_UDUMP, value=val):
				self.call_cfunction("__udump", [None])

			case Token(type=TokenTypes.OP_HEXDUMP, value=val):
				self.call_cfunction("__hexdump", [None])

			case Token(type=TokenTypes.OP_EXIT, value=val):
				self.block("EXIT")
				self.asm2("mov", "rax", "60")
				self.asm2("mov", "rdi", f"{val}")
				self.asm("syscall")

			case Token(type=TokenTypes.OP_IF, value=val):
				self.block("IF")
				self.label(instruction.label())
				self.asm1("pop", "rax")
				self.asm2("test", "rax", "rax")
				self.asm1("jz", f"{val[1].label()}")

			case Token(type=TokenTypes.OP_ELSE, value=val):
				self.block("ELSE")
				self.asm1("jmp", f"{val[1].label()}")
				self.label(instruction.label())
				

			case Token(type=TokenTypes.OP_WHILE, value=val):
				self.block("WHILE")
				self.label(instruction.label())

			case Token(type=TokenTypes.OP_DO, value=val):
				self.block("DO")
				self.label(instruction.label())
				self.asm1("pop", "rax")
				self.asm2("test", "rax", "rax")
				self.asm1("jmp", f"{val[1].label()}")
				
			case Token(type=TokenTypes.OP_END, value=val):
				self.block("END")
				if val[2].type in [TokenTypes.OP_WHILE,]:
					self.asm1("jmp", f"{val[2].label()}")

			case Token(type=TokenTypes.OP_MEM, value=val):
				self.block("mem")
				self.asm1("push", "mem")
				
			case Token(type=TokenTypes.OP_STORE, value=val):
				self.block("store")
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm2("mov", "byte [rax]", "bl")

			case Token(type=TokenTypes.OP_LOAD, value=val):
				self.block("load")
				self.asm1("pop", "rbx")
				self.asm1("pop", "rax")
				self.asm2("mov", "bl", "byte [rax]")
				self.asm1("push", "rbx")

			case _:
				print(instruction)
				raise Exception
		return 0

	def close(self):
		self.block("MEMORY")
		self.asm1("segment", ".bss")
		self.label("mem")
		self.asm1("resb", f"{MEM_CAPACITY}")

		align = max([j.size for i in self._code for j in i])
		for i in self._code:
			for j in i:
				self.buffer.write(j % align)
			self.buffer.write('\n')

class Interpreter:
	def __init__(self):
		self.queue = LifoQueue(-1)

	def close(self):
		pass
	def push(self, v: Any): self.queue.put(v)
	def pop(self) -> Any: return self.queue.get_nowait()
	def step(self, instruction: Token):
		assert TokenTypes.OP_COUNT.value == 20, f"Not all operators are handled {TokenTypes.OP_COUNT.value}"
		match instruction:
			case Token(type=TokenTypes.OP_PUSH, value=val):
				self.queue.put(val)
			case Token(type=TokenTypes.OP_POP, value=val):
				self.pop()
			case Token(type=TokenTypes.OP_DUP, value=val):
				a = self.pop()
				self.push(a)
				self.push(a)
			case Token(type=TokenTypes.OP_PLUS, value=val):
				self.push(self.pop() + self.pop())
			case Token(type=TokenTypes.OP_MUL, value=val):
				self.push(self.pop() * self.pop())
			case Token(type=TokenTypes.OP_DIV, value=val):
				self.push(self.pop() // self.pop())

			case Token(type=TokenTypes.OP_EQ, value=val):
				self.push(int(self.pop() == self.pop()))
			case Token(type=TokenTypes.OP_NE, value=val):
				self.push(int(self.pop() != self.pop()))
			case Token(type=TokenTypes.OP_GT, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(a > b))
			case Token(type=TokenTypes.OP_GE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(a >= b))
			case Token(type=TokenTypes.OP_LT, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(a < b))
			case Token(type=TokenTypes.OP_LE, value=val):
				a = self.pop()
				b = self.pop()
				self.push(int(a <= b))
			case Token(type=TokenTypes.OP_MINUS, value=val):
				a = self.pop()
				b = self.pop()
				self.push(b - a)
			case Token(type=TokenTypes.OP_DUMP, value=val):
				print(self.pop())
			case Token(type=TokenTypes.OP_IF, value=val):
				a = self.pop()
				if a == 0:
					return val[0]
			case Token(type=TokenTypes.OP_ELSE, value=val):
				return val[0]
			case Token(type=TokenTypes.OP_WHILE, value=val):
				pass
			case Token(type=TokenTypes.OP_DO, value=val):
				a = self.pop()
				if a == 0:
					return val[0]
			case Token(type=TokenTypes.OP_END, value=val):
				if val[2].type in [TokenTypes.OP_WHILE,]:
					return -(val[0] + val[2].value[0])
			case Token(type=TokenTypes.OP_EXIT, value=val):
				exit(val)
			case _:
				raise NotImplemented(instruction)
		return 0

class Program:
	def __init__(self, engine=None):
		self.instructions: list[Token] = []
		self.engine = engine
		self.pointer = 0

	@classmethod
	def fromfile(cls, path):
		with open(path, 'r') as f:
			return cls.frombuffer(f)

	@classmethod
	def frombuffer(cls, buffer):
		assert TokenTypes.OP_COUNT.value == 25, f"Not all operators are handled {TokenTypes.OP_COUNT.value}"
		tokens = tokenize.generate_tokens(buffer.readline)
		self = cls()
		for token in tokens:
			match token:
				case tokenize.TokenInfo(type=tokenize.NUMBER, string=s):
					self.add(PUSH(int(s), info=token))
				case tokenize.TokenInfo(type=tokenize.OP, string=s):
					if s == '+': self.add(PLUS(info=token))
					elif s == '-': self.add(MINUS(info=token))
					elif s == '*': self.add(MUL(info=token))
					elif s == '/': self.add(DIV(info=token))
					elif s == '==': self.add(EQ(info=token))
					elif s == '!=': self.add(NE(info=token))
					elif s == '>': self.add(GT(info=token))
					elif s == '>=': self.add(GE(info=token))
					elif s == '<': self.add(LT(info=token))
					elif s == '<=': self.add(LE(info=token))
					else: raise UnknownToken(token)
				case tokenize.TokenInfo(type=tokenize.NAME, string=s):
					if s == 'dump': self.add(DUMP(info=token))
					elif s == 'udump': self.add(UDUMP(info=token))
					elif s == 'hexdump': self.add(HEXDUMP(info=token))
					elif s == 'dup': self.add(DUP(info=token))
					elif s == 'exit': self.add(EXIT(info=token))
					elif s == 'if': self.add(IF(info=token))
					elif s == 'else': self.add(ELSE(info=token))
					elif s == 'while': self.add(WHILE(info=token))
					elif s == 'do': self.add(DO(info=token))
					elif s == 'end': self.add(END(info=token))
					elif s == 'mem': self.add(MEM(info=token))
					elif s == 'store': self.add(STORE(info=token))
					elif s == 'load': self.add(LOAD(info=token))
					else: raise UnknownToken(token)
				case tokenize.TokenInfo(type=tokenize.STRING, string=s):
					raise UnknownToken(token)
		self.process_flow_control(iter(self.instructions))
#		for token in self.instructions:
#			print(token)
#		exit()
		return self
	
	def process_flow_control(self, iterator):
		assert TokenTypes.OP_COUNT.value == 25, f"Not all operators are handled {TokenTypes.OP_COUNT.value}"
		stack = []
		for pos, token in enumerate(iterator):
			if token.type == TokenTypes.OP_IF:
				stack.append((pos, token))
			elif token.type == TokenTypes.OP_ELSE:
				p,t = stack.pop(-1)
				if t.type not in [TokenTypes.OP_IF]:
					raise InvalidSyntax(t.info)
				t.value = pos - p, token, None
				token.value = pos - p, t, t
				stack.append((pos, token))
			elif token.type == TokenTypes.OP_END:
				p,t = stack.pop(-1)
				if t.type not in [TokenTypes.OP_IF, TokenTypes.OP_ELSE, TokenTypes.OP_DO]:
					raise InvalidSyntax(t.info)
				t.value = pos - p, token, t.value[2] if t.value else t
				token.value = pos - p, t, t.value[2]
			elif token.type == TokenTypes.OP_WHILE:
				stack.append((pos, token))
			elif token.type == TokenTypes.OP_DO:
				p,t = stack.pop(-1)
				if t.type not in [TokenTypes.OP_WHILE]:
					raise InvalidSyntax(t.info)
				t.value = pos - p, token, None
				token.value = pos - p, t, t
				stack.append((pos, token))
		if stack:
			raise InvalidSyntax(stack[0][1].info)

	def add(self, token: Token) -> 'Program':
		self.instructions.append(token)
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
	parser.add_argument("engine", choices=["interpret", "compile", "fclean"])
	parser.add_argument("-s", "--source")
	parser.add_argument("-o", "--output", default="a.out")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-e", "--exec", action="store_true")
	args = parser.parse_args()

	if args.source:
		with open(args.source, 'r') as f:
			try:
				p = Program.frombuffer(f)
			except (UnknownToken, InvalidSyntax) as e:
				token: tokenize.TokenInfo = e.args[0]
				print(f"\033[31mError: line {token.start[0]}: {e.__class__.__name__}:\033[0m\n")
				print(token.line, end='')
				print(f"{'': <{token.start[1]}}{'^':^<{token.end[1] - token.start[1]}}")
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
