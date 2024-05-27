from queue import Queue, Empty
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Optional, Any

import subprocess
import sys
import os
from pathlib import Path
import shutil

import argparse
import tokenize


def iota(reset=False, *, v=[-1]):
	if reset:
		v[0] = -1
	v[0] += 1
	return v[0]

class UnknownToken(Exception):
	pass

class TokenTypes(Enum):
	OP_PUSH		= iota(True)
	OP_POP		= iota()
	OP_PLUS		= iota()
	OP_MINUS	= iota()
	OP_DUMP		= iota()
	OP_EXIT		= iota()
	OP_COUNT	= iota()

class Token:
	__slots__ = ("type", "value")

	type: TokenTypes
	value: Any

	def __init__(self, type: TokenTypes, value: Any=None):
		self.value = value
		self.type = type

	def __repr__(self):
		return f"{self.type}{f" {self.value}" if self.value else ""}"

def PUSH(val: Any) -> Token:
	return Token(TokenTypes.OP_PUSH, val)

def POP() -> Token:
	return Token(TokenTypes.OP_POP)

def PLUS() -> Token:
	return Token(TokenTypes.OP_PLUS)

def MINUS() -> Token:
	return Token(TokenTypes.OP_MINUS)

def DUMP() -> Token:
	return Token(TokenTypes.OP_DUMP)

def EXIT(code: int=0) -> Token:
	return Token(TokenTypes.OP_EXIT, value=code)

def iterqueue(q):
	while True:
		try:
			yield q.get_nowait()
		except Empty:
			return


class Compiler:
	def __init__(self, buffer):
		self.buffer = buffer
		self.buffer.write("extern __dump\n")
		self.buffer.write("segment .data\n")
		self.buffer.write("segment .text\n")
		self.buffer.write("global _start\n")
		self.buffer.write("_start:\n")

	def call_cfunction(self, name: str, args: list[Any | None]):
		"""  x86 arg registers
		arg0 (%rdi)	arg1 (%rsi)	arg2 (%rdx)	arg3 (%r10)	arg4 (%r8)	arg5 (%r9)
		"""

		self.buffer.write(f"  ; {name} {args}\n")
		registers = ["rdi", "rsi", "rdx", "r10", "r8", "r9"]
		if len(args) < 7:
			for reg, arg in reversed([*zip(registers, args)]):
				if arg is None:
					self.buffer.write(f"  pop  {reg}\n")
				else:
					self.buffer.write(f"  mov  {reg},{arg}\n")

		self.buffer.write(f"  push rbp\n")
		self.buffer.write(f"  mov  rbp,rsp\n")
		self.buffer.write(f"  call __dump\n")
		self.buffer.write(f"  pop  rbp\n")

	def step(self, instruction: Token):
		assert TokenTypes.OP_COUNT.value == 6, "Not all operators are handled"
		match instruction:
			case Token(type=TokenTypes.OP_PUSH, value=val):
				self.buffer.write(f"  ; push {val}\n")
				self.buffer.write(f"  push {val:.0f}\n")
			case Token(type=TokenTypes.OP_POP, value=val):
				self.buffer.write(f"  ; pop\n")
				self.buffer.write(f"  ; NOT YET IMPLEMENTED\n")
			case Token(type=TokenTypes.OP_PLUS, value=val):
				self.buffer.write(f"  ; pop\n")
				self.buffer.write(f"  pop  rax\n")
				self.buffer.write(f"  pop  rbx\n")
				self.buffer.write(f"  add  rax,rbx\n")
				self.buffer.write(f"  push rax\n")
			case Token(type=TokenTypes.OP_MINUS, value=val):
				self.buffer.write(f"  ; pop\n")
				self.buffer.write(f"  pop  rax\n")
				self.buffer.write(f"  pop  rbx\n")
				self.buffer.write(f"  sub  rbx,rax\n")
				self.buffer.write(f"  push rbx\n")
			case Token(type=TokenTypes.OP_DUMP, value=val):
				self.call_cfunction("__dump", [None])
			case Token(type=TokenTypes.OP_EXIT, value=val):
				self.buffer.write(f"  ; EXIT\n")
				self.buffer.write(f"  mov  rax,60\n")
				self.buffer.write(f"  mov  rdi,{val}\n")
				self.buffer.write(f"  syscall\n")
			case _:
				raise NotImplemented(instruction)

class Interpreter:
	def __init__(self):
		self.queue = Queue(-1)

	def push(self, v: Any): self.queue.put(v)
	def pop(self) -> Any: return self.queue.get_nowait()
	def step(self, instruction: Token):
		assert TokenTypes.OP_COUNT.value == 6, "Not all operators are handled"
		match instruction:
			case Token(type=TokenTypes.OP_PUSH, value=val):
				self.queue.put(val)
			case Token(type=TokenTypes.OP_POP, value=val):
				self.pop()
			case Token(type=TokenTypes.OP_PLUS, value=val):
				self.queue.put(self.pop() + self.pop())
			case Token(type=TokenTypes.OP_MINUS, value=val):
				a = self.pop()
				b = self.pop()
				self.queue.put(b - a)
			case Token(type=TokenTypes.OP_DUMP, value=val):
				print(self.pop())
#				for i in iterqueue(self.queue):
#					print(i)
			case Token(type=TokenTypes.OP_EXIT, value=val):
				exit(val)
			case _:
				raise NotImplemented(instruction)

class Program:
	def __init__(self, engine=None):
		self.queue = Queue(-1)
		self.engine = engine

	@classmethod
	def fromfile(cls, path):
		with open(path, 'r') as f:
			return cls.frombuffer(f)

	@classmethod
	def frombuffer(cls, buffer):
		tokens = tokenize.generate_tokens(buffer.readline)
		self = cls()
		for token in tokens:
			match token:
				case tokenize.TokenInfo(type=tokenize.NUMBER, string=s):
					self.add(PUSH(int(s)))
				case tokenize.TokenInfo(type=tokenize.OP, string=s):
					if s == '+': self.add(PLUS())
					elif s == '-': self.add(MINUS())
					else: raise UnknownToken(token)
				case tokenize.TokenInfo(type=tokenize.NAME, string=s):
					if s == 'dump': self.add(DUMP())
					elif s == 'exit': self.add(EXIT())
					else: raise UnknownToken(token)
				case tokenize.TokenInfo(type=tokenize.STRING, string=s):
					raise UnknownToken(token)
		return self
			

	def add(self, token: Token) -> 'Program':
		self.queue.put(token)
		return self

	def run(self):
		if self.engine is None:
			raise ValueError("Add engine before running")
		for i in iterqueue(self.queue):
			self.engine.step(i)

def callcmd(cmd, verbose=False):
	if verbose:
		print("BUILD:", cmd)
		return subprocess.call(cmd)
	else:
		return subprocess.call(cmd, stdout=subprocess.DEVNULL)



def main(ac: int, av: list[str]):
	parser = argparse.ArgumentParser(prog="slang", description="A stack based language written in python")
	parser.add_argument("engine", choices=["interpret", "compile", "fclean"])
	parser.add_argument("-s", "--source")
	parser.add_argument("-o", "--output", default="a.out")
	parser.add_argument("-v", "--verbose", action="store_true")
	args = parser.parse_args()

	if args.source:
		with open(args.source, 'r') as f:
			try:
				p = Program.frombuffer(f)
			except UnknownToken as e:
				token: tokenize.TokenInfo = e.args[0]
				print("\033[31mError: Unknown token:\033[0m\n")
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

		callcmd(["make", "-C", "src/cfunc/"], verbose=args.verbose)
		callcmd(["nasm", "-f", "elf64", "objs/intermediary.asm", "-o", "objs/intermediary.o"], verbose=args.verbose)
		callcmd(["ld",
				 "src/cfunc/objs/dump.o",
				 "objs/intermediary.o",
				 "-lc",
				 "-I",
				 "/lib64/ld-linux-x86-64.so.2",
		   		 "-o",
		   		 args.output
			],
			verbose=args.verbose
		)
