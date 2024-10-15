from typing import TextIO, cast
from pathlib import Path

import subprocess

from .lexer import Program
from .typechecker import TypeChecker
from .engine import Compiler
from .classes import (
	Error,
	Missing,
	Errors,
)

class MakeException(Exception): pass
class NASMException(Exception): pass
class LinkerException(Exception): pass


def align_tabs(s: str, n: int, ts: int=4):
	j = 0
	k = 0
	for i,c in enumerate(s[:n]):
		if c == '\t':
			k += ts if j == 0 else -j % ts
			j = 0
		else:
			j = (j + 1) % ts
			k += 1
	return k

def get_line(f, n):
	f.seek(0)
	for i in zip(range(n - 1), f):
		pass
	return f.readline()

def callcmd(cmd, verbose=False, devnull=True):
	if verbose:
		print("CMD:", cmd)
	if devnull:
		return subprocess.call(cmd, stdout=subprocess.DEVNULL)
	else:
		return subprocess.call(cmd)

def trace(e: Errors.TypeCheckerException):
	if len(e.args) > 2 and e.args[2] is not None:
		trace(cast(Errors.TypeCheckerException, e.args[2]))
	
	for i in e.args[0][::-1]:
		a, b = i.node.range.start_point, i.node.range.end_point
		print(f"\033[31mError:\033[0m", end='') # {file} line {start[0] + 1}\033[0m")
		print(f"line {i.node.range.start_point[0] + 1}")
		file = i.node.file
		if not file:
			raise ValueError("No file information")
		with open(file, 'r') as f:
			line = get_line(f, a[0] + 1)
		start_x = align_tabs(line, a[1])
		end_x = align_tabs(line, b[1])
		print(line.expandtabs(4))
		print(f'{'':>{start_x}}{'':^>{end_x - start_x}}')
	print(e.args[1])


def compile(*, source: str | Path,
		 output: str | Path | TextIO,
		 temp: str | Path,
		 verbose: bool=False,
		 includes: list[str],
		 execution: bool=False,
		 argv: list[str]) -> int:
	try:
		p = Program(includes=[Path(i) for i in includes])
		tc = TypeChecker()
		l = []
		for i in p.parse(source):
			tc.check(i)
			l.append(i)

		with open(".gdbinit", 'w') as f:
			pass
		objs = Path(temp)
		if not objs.exists():
			objs.mkdir()
		compiler = Compiler(root=True)
		compiler.run(l)
		with open(objs / 'intermediary.asm', 'w') as f:
			compiler.close(f)
		

		if e:=callcmd(["make", "-C", "src/cfunc/"], verbose=verbose, devnull=True):
			raise MakeException(e)
		if e:=callcmd(["nasm",
					"-f", "elf64",
					"-F", "dwarf", "-g", 
					"objs/intermediary.asm",
					"-o", "objs/intermediary.o"], verbose=verbose, devnull=True):
			raise NASMException(e)
		if e:=callcmd(["ld",
					"src/cfunc/objs/dump.o",
					"src/cfunc/objs/memory.o",
					"objs/intermediary.o",
					"-lc",
					"-I", "/lib64/ld-linux-x86-64.so.2",
					"-o", output ], verbose=verbose, devnull=True):
			raise LinkerException(e)

		if execution:
			return callcmd([f"./{output}", *argv], verbose=verbose, devnull=False)
		return 0



	except (Missing, Error) as e:
		text = e.args[0].text.decode() if e.args[0].text else ''
		start = e.args[1].range.start_point
		end = e.args[1].range.end_point
		msg = e.args[3]
		file = e.args[2][-1]

		print(f"\033[31mError:\033[0m", end='') # {file} line {start[0] + 1}\033[0m")
		for i in e.args[2]:
			print(f"\nNOTE: file {i}", end='')
		print(f"line {start[0] + 1}")
		print(msg)
		raise e
		with open(e.args[2][-1], 'r') as f:
			line = get_line(f, start[0]).expandtabs(4)
		start_x = align_tabs(line, start[1])
		end_x = align_tabs(line, end[1])
		print(line)
		print(f'{'':>{start_x}}{'':^>{end_x - start_x}}')

	except Errors.LexerException as e:
		text = e.args[0].text.decode() if e.args[0].text else ''
		start = e.args[0].range.start_point
		end = e.args[0].range.end_point
		msg = e.args[2]

		print(f"\033[31mError:\033[0m", end='') # {file} line {start[0] + 1}\033[0m")
		for i in e.args[1]:
			print(f"\nNOTE: file {i}", end='')
		print(f"line {start[0] + 1}")
		print(msg)
		line = get_line(e.args[-1], start[0])
		start_x = align_tabs(line, start[1])
		end_x = align_tabs(line, end[1])
		print(line)
		print(f'{'':>{start_x}}{'':^>{end_x - start_x}}')

	except Errors.TypeCheckerException as e:
		trace(e)

	return -1
