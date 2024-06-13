import subprocess
import sys
from pathlib import Path
import argparse
from io import StringIO
import difflib

if __name__ == '__main__':
	from .lang import (
		compile, interpret, fclean, callcmd,
		MakeException, NASMException, LinkerException
	)
	from .engine import Interpreter, Compiler
	import sys

	parser = argparse.ArgumentParser(prog="slang", description="A stack based language written in python")
	sub = parser.add_subparsers(dest='engine')

	sub_compile = sub.add_parser("compile", help="Compile the provided source file to an elf64 executable")
	sub_interpret = sub.add_parser("interpret", help="Executes the provided source file using an interpreter")
	sub_lex = sub.add_parser("lex", help="Lexes the provided source file. TODO: create a language server")
	sub_fclean = sub.add_parser("fclean", help="Cleans the dump files used for compilation. (and temporarily the objs/ folder)")


	sub_compile.add_argument('-s', '--source', required=True,
						  help="The source code")
	sub_compile.add_argument('-t', '--temp', default='objs/',
						  help="The temp folder for intemediary asm and object files.")
	sub_compile.add_argument('-o', '--output', default='a.out',
						  help="The output executable file")
	sub_compile.add_argument('-v', '--verbose', action="store_true", default=False,
						  help="Displays every command executed")
	sub_compile.add_argument('-e', '--exec', action="store_true",
						  help="Automatically executes the resulting executable")
	sub_compile.add_argument('-I', '--include', nargs='*', default=[], action='extend',
						  help="Adds every -I --include folder provided to the search path for includes, if a file is present in multiple path, the first one is used, by default, the source folder, cwd and compiler path are added.")
	sub_compile.add_argument('-A', '--argv', nargs='*', default=[], action='extend',
						  help="If -e --exec flag, adds every value provided as an argv, otherwise ignored")


	sub_interpret.add_argument('-s', '--source', required=True,
						  help="The source code")
	sub_interpret.add_argument('-o', '--output', default='stdout',
						  help="Adds every -I --include folder provided to the search path for includes, if a file is present in multiple path, the first one is used")
	sub_interpret.add_argument('-I', '--include', nargs='*', default=[], action='extend',
						  help="Adds every -I --include folder provided to the search path for includes, if a file is present in multiple path, the first one is used")
	sub_interpret.add_argument('-A', '--argv', nargs='*', default=[], action='extend',
						  help="Adds every value provided as an argv, by default the source's basename is prepended as a path")

	sub_lex.add_argument('-s', '--source', required=True,
						  help="The source code")

	args = parser.parse_args()

	sub_fclean.add_argument('-v', '--verbose', action="store_true",
						  help="Displays every command executed")

	if args.engine == 'fclean':
		fclean(verbose=args.verbose)

#	if args.engine == 'test':
#		for file in Path(args.source).glob('*.pyslang') if Path(args.source).is_dir() else [Path(args.source)]:
#			print(f"Test [{file}]")
#			try:
#				code = main(source=file, output=file.with_suffix(''), engine=Compiler, includes=args.include)
#			except (MakeException, NASMException, LinkerException) as e:
#				pass
#
#			sio = StringIO()
#			sys.modules['__main__'].argv = [f"./{file.with_suffix('')}", *args.argv] # pyright: ignore
#			interp_code = main(source=file, output=sio, engine=Interpreter, includes=args.include)
#			sio.seek(0)
#
#			p = subprocess.run([f"./{file.with_suffix('')}", *args.argv], stdout=subprocess.PIPE)
#			compil_code = p.returncode
#
#			if interp_code != compil_code:
#				print(f"[{file}] \033[31mFailure\033[0m: Different return code")
#			compil_bytes = p.stdout
#			interp_bytes = sio.read().encode('utf8')
#
#			if interp_bytes != compil_bytes:
#				print(f"[{file}] \033[31mFailure\033[0m: Different content")
#				
##				print(interp_bytes.decode('utf8').split('\n'))
##				print(compil_bytes.decode('utf8').split('\n'))
#				for i in difflib.unified_diff(interp_bytes.decode('utf8').split('\n'), compil_bytes.decode('utf8').split('\n')):
#					print(i)
#			else:
#				print(f"[{file}] \033[32mSuccess\033[0m")
#
#		exit(0)

	if args.engine == 'interpret':
		interpret(source=args.source, includes=args.include, argv=[f"./{Path(args.source).stem}", *args.argv], output=args.output)
	elif args.engine == 'compile':
		if args.argv and not args.exec:
			parser.error("-A --argv requires -e --exec")
		try:
			compile(source=args.source, output=args.output, temp=args.temp, verbose=args.verbose, includes=args.include, execution=args.exec, argv=args.argv)
		except (MakeException, NASMException, LinkerException) as e:
			print(e.__qualname__, e.args[0])
