import tester
import argparse
from io import BytesIO


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="instruction")

	sread = sub.add_parser("test")
	sread.add_argument('-s', '--source', required=True, nargs='*')

	swrite = sub.add_parser("save")
	swrite.add_argument('-o', '--output', required=True)
	swrite.add_argument('-s', '--source', required=True)
	swrite.add_argument('-a', '--args', required=True, nargs='*')
	swrite.add_argument('-A', '--cargs', nargs='*', default=[])
	swrite.add_argument('--stdin', default=None)

	args = parser.parse_args()

	if args.instruction == 'test':
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
				  f"-stests/{args.source}",
				  "--exec",
				  "-I./src/std/",
				  *[ i for i in args.cargs ],
				  *[ f"-A{i}" for i in args.args ],
				  ],
				  args.stdin)
			tester.save(buffer,
				  ["python", "-m", "src", "interpret",
				  f"-stests/{args.source}",
				  "-I./src/std/",
				  *[ i for i in args.cargs ],
				  *[ f"-A{i}" for i in args.args ],
				  ],
				  args.stdin)
			buffer.seek(0)

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
