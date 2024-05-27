from queue import Queue, Empty
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Optional, Any

import subprocess
import sys
import os

def iota(reset=False, *, v=[-1]):
	if reset:
		v[0] = -1
	v[0] += 1
	return v[0]
	
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
				self.buffer.write(f"  sub  rax,rbx\n")
				self.buffer.write(f"  push rax\n")
			case Token(type=TokenTypes.OP_DUMP, value=val):
				self.buffer.write(f"  ; dump\n")
				self.buffer.write(f"  pop   rdi\n")
				self.buffer.write(f"  push  rbp\n")
				self.buffer.write(f"  mov   rbp,rsp\n")
				self.buffer.write(f"  call  __dump\n")
				self.buffer.write(f"  pop   rbp\n")
			case Token(type=TokenTypes.OP_EXIT, value=val):
				self.buffer.write(f"  ; EXIT\n")
				self.buffer.write(f"  mov  rax,60\n")
				self.buffer.write(f"  mov  rdi,{val}\n")
				self.buffer.write(f"  syscall\n")
			case _:
				raise NotImplemented(instruction)

class Simulator:
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
				for i in iterqueue(self.queue):
					print(i)
			case Token(type=TokenTypes.OP_EXIT, value=val):
				exit(val)
			case _:
				raise NotImplemented(instruction)

class Program:
	def __init__(self, engine=None):
		self.queue = Queue(-1)
		self.engine = engine

	def add(self, token: Token) -> 'Program':
		self.queue.put(token)
		return self

	def run(self):
		if self.engine is None:
			raise ValueError("Add engine before running")
		for i in iterqueue(self.queue):
			self.engine.step(i)

def main(ac: int, av: list[str]):
	if ac == 1:
		print(f"Usage: {av[0]} [0 || 1]")
		return

	p = (Program()
		.add(PUSH(4))
		.add(PUSH(4))
		.add(PLUS())
		.add(DUMP())
		.add(EXIT())
	)
	if av[1] == '0':
		p.engine = Simulator()
		p.run()
	elif av[1] == '1':
		with open("output.asm", 'w') as f:
			p.engine = Compiler(f)
			p.run()
		subprocess.call(["gcc", "-c", "src/dump.c", "-o", "src/dump.o", "-lasm", "-g", "-ggdb"])
		subprocess.call(["nasm", "-f", "elf64", "output.asm"])
		subprocess.call(["ld", "src/dump.o", "output.o", "-lc", "-I", "/lib64/ld-linux-x86-64.so.2"])
