import subprocess
import sys
from pathlib import Path
import argparse
from io import StringIO
import difflib

if __name__ == '__main__':
	from .lang import (
		main, fclean, callcmd,
		MakeException, NASMException, LinkerException
	)
	from .engine import Interpreter, Compiler
	import sys

	parser = argparse.ArgumentParser(prog="slang", description="A stack based language written in python")
	parser.add_argument("engine", choices=["interpret", "compile", "fclean", "lex", "test"])
	parser.add_argument("-s", "--source", required=True)
	parser.add_argument("-o", "--output", default=None)
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-e", "--exec", action="store_true")
	parser.add_argument("-I", "--include", nargs='*', default=[])
	parser.add_argument("-A", "--argv", nargs='*', default=[])
	args = parser.parse_args()


	if args.engine == 'fclean':
		fclean(verbose=args.verbose)

	if args.engine == 'test':
		for file in Path(args.source).glob('*.pyslang') if Path(args.source).is_dir() else [Path(args.source)]:
			print(f"Test [{file}]")
			try:
				code = main(source=file, output=file.with_suffix(''), engine=Compiler, includes=args.include)
			except (MakeException, NASMException, LinkerException) as e:
				pass

			sio = StringIO()
			sys.modules['__main__'].argv = [f"./{file.with_suffix('')}", *args.argv] # pyright: ignore
			interp_code = main(source=file, output=sio, engine=Interpreter, includes=args.include)
			sio.seek(0)

			p = subprocess.run([f"./{file.with_suffix('')}", *args.argv], stdout=subprocess.PIPE)
			compil_code = p.returncode

			if interp_code != compil_code:
				print(f"[{file}] \033[31mFailure\033[0m: Different return code")
			compil_bytes = p.stdout
			interp_bytes = sio.read().encode('utf8')

			if interp_bytes != compil_bytes:
				print(f"[{file}] \033[31mFailure\033[0m: Different content")
				
#				print(interp_bytes.decode('utf8').split('\n'))
#				print(compil_bytes.decode('utf8').split('\n'))
				for i in difflib.unified_diff(interp_bytes.decode('utf8').split('\n'), compil_bytes.decode('utf8').split('\n')):
					print(i)
			else:
				print(f"[{file}] \033[32mSuccess\033[0m")

		exit(0)

	engine = None
	if args.engine == 'interpret':
		engine = Interpreter
		sys.modules['__main__'].argv = [args.source, *args.argv] # pyright: ignore
		args.output=sys.stdout
	elif args.engine == 'compile':
		engine = Compiler
		if args.output is None:
			args.output='a.out'
	else:
		raise Exception("Unreachable")
	try:
		exit(main(source=args.source, output=args.output, engine=engine, execution=args.exec, includes=args.include))
	except (MakeException, NASMException, LinkerException) as e:
		print(e.__qualname__, e.args[0])
