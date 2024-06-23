from .lexer import (TokenInfo, Token)
import sys

from .errors import (
	Reporting,
)

def warn(error):
	token: TokenInfo = error.args[0]
	msg = error.args[1]
	if len(error.args) == 3:
		warn(error.args[2])
	if token.parent:
		note_parent(token.parent)
	print(f"\033[33mWarning: {token.file} line {token.start[0] + 1}: {error.__class__.__name__}:\033[0m\n", file=sys.stderr)
	print(token.error(), file=sys.stderr)
	print(msg, file=sys.stderr)


def note_parent(token):
	if token.parent:
		note_parent(token.parent)
	print(f"\033[32mNOTE\033[0m From {token.file} line {token.start[0] + 1}:", file=sys.stderr)
	print(token.error(), file=sys.stderr)
		
def trace(error):
	token: TokenInfo = error.args[0]
	msg = error.args[1]
	if len(error.args) == 3:
		d = trace(error.args[2])
	if token.parent:
		note_parent(token.parent)
	if type(error) == Reporting:
		print(f"{token.file} line {token.start[0] + 1}:", file=sys.stderr)
	else:
		print(f"\033[31mError: {token.file} line {token.start[0] + 1}: {error.__class__.__name__}:\033[0m\n", file=sys.stderr, end='')
	print(token.error(), file=sys.stderr)
	if msg:
		print(msg, file=sys.stderr, end='\n\n')
#	raise error

