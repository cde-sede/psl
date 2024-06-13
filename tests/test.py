import tester
import argparse
from io import BytesIO
from pathlib import Path


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="instruction")

	stest = sub.add_parser("test")
	stest.add_argument('-s', '--source', required=True, nargs='*', action='extend')

	sshow = sub.add_parser("show")
	sshow.add_argument('-s', '--source', required=True, nargs='*', action='extend')

	ssave = sub.add_parser("save")
	ssave.add_argument('-o', '--output', required=True)
	ssave.add_argument('-s', '--source', required=True)
	ssave.add_argument('-a', '--args', required=True, nargs='*', action='extend')
	ssave.add_argument('-A', '--cargs', nargs='*', default=[])
	ssave.add_argument('--stdin', default=None)

	args = parser.parse_args()

	if args.instruction == 'show':
		for s in args.source:
			with open(s, 'rb') as f:
				retc, haltedc, outc, errc, failurec = tester.test(f)
				reti, haltedi, outi, erri, failurei = tester.test(f)
				print('---------------')
				print(f"{retc} {reti}")
				print('---------------')
				print(f"{haltedc} {haltedi}")
				print('---------------')
				print(f"{outc} {outi}")
				print('---------------')
				print(f"{errc} {erri}")
				print('---------------')
				print(f"{failurec} {failurei}")
	if args.instruction == 'test':
		print(args)
		for s in args.source:
			with open(s, 'rb') as f:
				retc, haltedc, outc, errc, failurec = tester.test(f)
				reti, haltedi, outi, erri, failurei = tester.test(f)
				failure = 0
				if retc != reti:
					print(f"\033[31mRETURNCODE\033[0m {retc} != {reti}")
					failure = 1
				if haltedc != haltedi:
					print(f"\033[31mHALTED\033[0m {haltedc} != {haltedi}")
					failure = 1
				if outc != outi:
					print(f"\033[31mSTDOUT\033[0m {outc} != {outi}")
					failure = 1
				if errc != erri:
					print(f"\033[31mSTDERR\033[0m {errc} != {erri}")
					failure = 1
				if failurec != failurei:
					print(f"\033[31mFAILURE\033[0m {failurec} != {failurei}")
					failure = 1

	elif args.instruction == 'save':
		buffer = BytesIO()
		with open(args.output, 'wb') as buffer:
			tester.save(buffer,
				  ["python", "-m", "src", "compile",
				  f"-s{args.source}",
				  f"-o{Path(args.source).stem}",
				  "--exec",
				  "-I./src/std/",
				  *[ i for i in args.cargs ],
				  *[ f"-A{i}" for i in args.args ],
				  ],
				  args.stdin)
			tester.save(buffer,
				  ["python", "-m", "src", "interpret",
				  f"-s{args.source}",
				  "-I./src/std/",
				  *[ i for i in args.cargs ],
				  *[ f"-A{i}" for i in args.args ],
				  ],
				  args.stdin)

#python
#test.py
#save
#-o tests/argv
#-a'python'
#-a'-m'
#-a'src'
#-a'compile'
#-a'-stests/argv.pyslang'
#-a'-I./src/std/'
#-a"-A'asdf'"
#-a'--exec'
