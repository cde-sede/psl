import tree_sitter_pyslang as pyslang

import tree_sitter
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any, Generator, Literal, cast as _cast
from copy import deepcopy
from itertools import batched
from pathlib import Path
import os

PYSLANG = tree_sitter.Language(pyslang.language())

from . import classes

from .classes import (
	CONSTANTS,
	Node,
	Type,
	Types,
	TokenTypes,
	Builtins,
	Syscalls,
	Operands,
	Token,
	Operand,
	Builtin,
	Word,
	Syscall,
	Include,
	Cast,
	Pair,
	Sizeof,
	Accessor,
	Definition,
	Memory,
	Macro,
	ASM,
	Proc,
	Procs,
	Struct,
	Field,
	With,
	Let,
	FlowControl,
	While,
	If,
	Elif,
	Else,
	Do,
	End,
	StackType,

	Error,
	Missing,
	Errors,
)

class Ignore(Exception):
	pass

def DO(node: Node):
	return Do(TokenTypes.DO, 'do', node)

def END(node: Node):
	return End(TokenTypes.END, 'END', node)

def unescape_string(s):
	return s.encode('latin-1', 'backslashreplace').decode('unicode-escape')

def NODE(n: tree_sitter.Node, file: str | Path | None):
	return Node(n, file)

class Program:
	def __init__(self, includes: Optional[list[str | Path]]=None):
		self.path: str | Path | None = None
		
		self.includes: list[Path] = [Path(classes.__file__).parent / 'std', Path(os.getcwd()), Path(__file__).parent] + ([Path(i) for i in includes] if includes else [])
		self.prog: list[Token] = []
		self.types: list[Type] = [
			Types.ANY,
			Types.BOOL,

			Types.CHAR,
			Types.SHORT,
			Types.INT,
			Types.LONG,

			Types.UCHAR,
			Types.USHORT,
			Types.UINT,
			Types.ULONG,

			Types.BYTE,
			Types.WORD,
			Types.DWORD,
			Types.QWORD,

			Types.PTR,
		]

		self.parser = tree_sitter.Parser()
		self.parser.language = PYSLANG

	def search_path(self, path, query) -> Path:
		if not path:
			raise FileNotFoundError(query)
		if Path(path[0], query).exists():
			return Path(path[0], query)
		return self.search_path(path[1:], query)


	def get_type(self, s: str, node: tree_sitter.Node) -> Type:
		t, l = {
			i.name: (i, len(i.name)) for i in self.types
		}.get(s.rstrip('*'), [None, None])

		if t is None or l is None:
			raise Errors.UnknownType(node, [], f'Unknown type {s}')

		while s[l:] and s[l:][0] == '*':
			t = Types.PTR[t]
			l += 1
		return t

	def tokenize(self, node: tree_sitter.Node):
		for i in node.children:
			try:
				yield self._tokenize(i)
			except Ignore: continue

	def parse(self, path):
		before = self.path
		self.path = path
		with open(path, 'rb') as f:
			tree = self.parser.parse(f.read())

			for i in self.traverse_tree(tree):
				if i is None:
					continue
				if i.is_error:
					raise Error(i, i, [path], "ERROR")
				if i.is_missing:
					raise Missing(i, i, [path], "MISSING")
		try:
			for i in self.tokenize(tree.root_node):
				yield(i)
		except Error as e:
			raise Error(e.args[0], e.args[1], [path] + e.args[2], e.args[3]) from e
		except Missing as e:
			raise Missing(e.args[0], e.args[1], [path] + e.args[2], e.args[3]) from e
		except Errors.LexerException as e:
			raise type(e)(e.args[0], [path] + e.args[1], e.args[2]) from e
		self.path = before

	def traverse_tree(self, tree: tree_sitter.Tree):
		cursor = tree.walk()

		visited_children = False
		while True:
			if not visited_children:
				yield cursor.node
				if not cursor.goto_first_child():
					visited_children = True
			elif cursor.goto_next_sibling():
				visited_children = False
			elif not cursor.goto_parent():
				break

	def safe_child(self, node, n) -> tree_sitter.Node:
		if n > node.child_count:
			raise Errors.MissingNode(node, [], "Missing node, either a lexer mistake or a ts error")
		r = node.child(n)
		assert r is not None, '???'
		return r

	def safe_named_child(self, node, n) -> tree_sitter.Node:
		if n > node.named_child_count:
			raise Errors.MissingNode(node, [], "Missing node, either a lexer mistake or a ts error")
		r = node.named_child(n)
		assert r is not None, '???'
		return r

	def parse_proc(self, ins: tree_sitter.Node) -> Proc:
		name = ins.child_by_field_name('name')
		assert name is not None, '???'
		assert name.text is not None, '???'
		args = self.safe_child(ins, 2)
		body = self.safe_child(ins, 3)
		out = None
		if body.type == 'out':
			out = body
			body = self.safe_child(ins, 4)
		return Proc(
				TokenTypes.PROC,
				ins.text.decode() if ins.text else '',
				NODE(ins, self.path),
				name=name.text.decode(),
				body=list(self.tokenize(body)),
				args=_cast(list[Pair], list(self.tokenize(args))),
				out=_cast(list[Cast], list(self.tokenize(out))) if out else [],
		)

	def parse_definitions(self, def_: tree_sitter.Node, node: tree_sitter.Node) -> Token:
		ins = self.safe_child(def_, 0)
		name = ins.child_by_field_name('name')
		assert name is not None, '???'
		assert name.type == 'identifier', '???'
		assert name.text is not None, '???'

		if ins.type == 'memory':
			body = self.safe_child(ins, 2)
			typ = None
			if body.type == 'cast':
				typ = body
				body = self.safe_child(ins, 3)
			cast_type = Types.ANY
			if typ:
				assert typ.text is not None, '???'
				type_ = typ.text.decode()
				assert type_ is not None, '???'
				cast_type = self.get_type(type_[1:], node)
			return Memory(
					TokenTypes.MEMORY,
					ins.text.decode() if ins.text else '',
					NODE(node, self.path),
					name=name.text.decode(),
					typ=cast_type,
					body=list(self.tokenize(body)),
			)
		elif ins.type == 'macro':
			body = self.safe_child(ins, 2)
			return Macro(
					TokenTypes.MACRO,
					ins.text.decode() if ins.text else '',
					NODE(node, self.path),
					name=name.text.decode(),
					body=list(self.tokenize(body))
			)
		elif ins.type == 'proc':
			return self.parse_proc(ins)
		elif ins.type == 'struct':
			struct_body = self.safe_child(ins, 2)
			typ = Type(name.text.decode(), 0)
			self.types.append(typ)


			fields: dict[str, Field] = {}
			methods: dict[str, Procs] = {}
			total_size = 0

			#print("Struct", name.text.decode())
			for i in struct_body.children:
				if i.type == 'pair':
					pair = self.parse_pair(i)
					fields[pair.name.text] = Field(
						TokenTypes.FIELD,
						i.text.decode() if i.text else '',
						NODE(i, self.path),
						pair,
						total_size,
						pair.cast.cast_type.size,
					)
					#print(fields[pair.name.text])
					total_size += pair.cast.cast_type.size # // CONSTANTS.PACK_ALIGN * CONSTANTS.PACK_ALIGN + (CONSTANTS.PACK_ALIGN if pair.cast.cast_type.size % CONSTANTS.PACK_ALIGN else 0)
				if i.type == 'proc':
					p = self.parse_proc(i)
					if methods.get(p.name, None):
						methods[p.name].procs.append(p)
					else:
						methods[p.name] = Procs(p.name, [p])

			#print("----------")

			typ.size = total_size

			return Struct(
					TokenTypes.STRUCT,
					ins.text.decode() if ins.text else '',
					NODE(node, self.path),
					name=name.text.decode(),
					fields=fields,
					methods=methods,
					typ=typ,
			)
		raise NotImplementedError(node)

	def parse_flow_control(self, def_: tree_sitter.Node, node: tree_sitter.Node) -> Token:
		type_ = self.safe_child(node, 0)
		name = self.safe_child(type_, 0)
		if name.text == b'while':
			cond = self.safe_child(def_, 1)
			do   = self.safe_child(def_, 2)
			body = self.safe_child(def_, 3)
			end  = self.safe_child(def_, 4)
			return While(
				TokenTypes.WHILE,
				'',
				NODE(def_, self.path),
				condition=list(self.tokenize(cond)),
				do=DO(NODE(do, self.path)),
				body=list(self.tokenize(body)),
				end=END(NODE(end, self.path)),
			)
		elif name.text == b'if':
			cond = self.safe_child(def_, 1)
			do   = self.safe_child(def_, 2)
			body = self.safe_child(def_, 3)
			n = 2
			elifs = []
			else_ = None
			while n < def_.named_child_count and def_.named_child(n).type == 'elif_statement': # pyright: ignore
				elif_ = self.safe_named_child(def_, n)
				econd = self.safe_child(elif_, 1)
				edo   = self.safe_child(elif_, 2)
				ebody = self.safe_child(elif_, 3)
				elifs.append(Elif(
					TokenTypes.ELIF,
					'',
					NODE(elif_, self.path),
					condition=list(self.tokenize(econd)),
					do=DO(NODE(edo, self.path)),
					body=list(self.tokenize(ebody)),
				))
				n += 1
			if n < def_.named_child_count and def_.named_child(n).type == 'else_statement': # pyright: ignore
				els_ = self.safe_named_child(def_, n)
				ebody = self.safe_child(els_, 1)
				else_ = Else(
					TokenTypes.ELSE,
					'',
					NODE(els_, self.path),
					body=list(self.tokenize(ebody))
				)
				n += 1
			end = self.safe_named_child(def_, n)
			return If(
				TokenTypes.IF,
				'',
				NODE(def_, self.path),
				condition=list(self.tokenize(cond)),
				do=DO(NODE(do, self.path)),
				body=list(self.tokenize(body)),
				elifs=elifs,
				else_=else_,
				end=END(NODE(end, self.path)),
			)
		elif name.text == b'with':
			var = list(self.tokenize(self.safe_child(def_, 1)))
			body = self.safe_child(def_, 3)
			assert all(map(lambda i: isinstance(i, Pair), var)), '???'
			v = _cast(list[Pair], var)
			return With(
					TokenTypes.WITH,
					def_.text.decode() if def_.text else '',
					NODE(node, self.path),
					variables=v,
					body=list(self.tokenize(body)),
			)
		elif name.text == b'let':
			var = list(self.tokenize(self.safe_child(def_, 1)))
			body = self.safe_child(def_, 3)
			assert all(map(lambda i: isinstance(i, Pair), var)), '???'
			v = _cast(list[Pair], var)
			return Let(
					TokenTypes.LET,
					def_.text.decode() if def_.text else '',
					NODE(node, self.path),
					variables=v,
					body=list(self.tokenize(body)),
			)
		raise NotImplementedError(node, name.text)

	def parse_definition(self, node: tree_sitter.Node) -> Token:
		def_ = self.safe_child(node, 0)
		if def_.type == 'definitions':
			return self.parse_definitions(def_, node)
		elif def_.type == 'flow_control':
			return self.parse_flow_control(def_, node)
		raise NotImplementedError(def_)


	def parse_include(self, node: tree_sitter.Node) -> Token:
		string = self.safe_child(node, 1)
		content = self.safe_child(string, 1)

		assert content.text is not None, '???'
		path = content.text.decode()
		fpath = self.search_path(self.includes, path)

		return Include(
				TokenTypes.INCLUDE,
				str(fpath),
				NODE(node, self.path),
				file=path,
				body=[i for i in self.parse(fpath) if i]
		)

	def parse_cast(self, node: tree_sitter.Node) -> Cast:
		assert node.text is not None, '???'
		type_ = node.text.decode()
		assert type_ is not None, '???'
		cast = self.get_type(type_[1:], node)
		return Cast(
			TokenTypes.CAST,
			type_,
			NODE(node, self.path),
			cast,
		)

	def parse_pair(self, node: tree_sitter.Node) -> Pair:
		name, cast = list(self.tokenize(node))
		assert isinstance(cast, Cast), '???'
		return Pair(
			TokenTypes.PAIR,
			node.text.decode() if node.text else '',
			NODE(node, self.path),
			name=name,
			cast=cast,
		)

	def parse_accessor(self, node: tree_sitter.Node) -> Token:
		assert node.text is not None, '???'
		text = node.text.decode()
		typ = int(text[0] == '!') # 0 == set, 1 == get
		text = text[1:]

		var, _, field = text.partition('.')
		if not var and not field:
			raise Errors.InvalidAccessor(node, [], 'Invalid accessor syntax')
		t = self.get_type(var, node)

		return Accessor(
			TokenTypes.ACCESSOR,
			node.text.decode(),
			NODE(node, self.path),
			typ,
			t,
			field,
		)

		return Token(TokenTypes.ACCESSOR, node.text.decode() if node.text else '', node)

	def parse_operand(self, node: tree_sitter.Node) -> Token:
		assert node.text is not None, '???'
		operand = node.text.decode()

		optype = {
			'+':  Operands.PLUS,
			'-':  Operands.MINUS,
			'*':  Operands.MUL,
			'/%': Operands.DIVMOD,
			'/':  Operands.DIV,
			'%':  Operands.MOD,
			'++': Operands.INC,
			'--': Operands.DEC,
			'<<': Operands.BLSH,
			'>>': Operands.BRSH,
			'&':  Operands.BAND,
			'|':  Operands.BOR,
			'^':  Operands.BXOR,
			'==': Operands.EQ,
			'!=': Operands.NE,
			'>':  Operands.GT,
			'>=': Operands.GE,
			'<':  Operands.LT,
			'<=': Operands.LE,
		}.get(operand, None)
		if optype is None:
			raise ValueError(node.start_point)

		return Operand(
			TokenTypes.OPERAND,
			operand,
			NODE(node, self.path),
			optype
		)

	def parse_sizeof(self, node: tree_sitter.Node) -> Token:
		assert node.text is not None, '???'
		type_ = node.text.decode()
		assert type_ is not None, '???'
		typ_ = self.get_type(type_[1:], node)
		assert typ_ is not None, 'TODO error reporting'
		return Sizeof(
			TokenTypes.SIZEOF,
			type_,
			NODE(node, self.path),
			typ_,
			typ_.size,
		)

	def parse_builtin(self, node: tree_sitter.Node) -> Token:
		sub = self.safe_child(node, 0)
		if sub.type == 'keyword' or sub.type == 'function_call':
			bkeyword = sub.text
			assert bkeyword is not None, '???'
			keyword = bkeyword.decode()
			e = Builtins[keyword.upper()]
			return Builtin(
					TokenTypes.BUILTIN,
					keyword,
					NODE(node, self.path),
					e,
					)
		elif sub.type == 'syscalls':
			bkeyword = sub.text
			assert bkeyword is not None, '???'
			keyword = bkeyword.decode()
			e = Syscalls[keyword.upper()]
			order = 1
			if keyword[0] == 'r':
				order = -1

			nargs = 0
			if keyword[-1].isdigit():
				nargs = int(keyword[-1])

			return Syscall(
					TokenTypes.SYSCALLS,
					keyword,
					NODE(node, self.path),
					order,
					nargs,
			)
			print(e)

		raise NotImplementedError(node)

	def parse_asm(self, node: tree_sitter.Node) -> Token:
		assert node.text is not None, '???'

		args = self.safe_child(node, 1)
		out = self.safe_child(node, 2)
		body = self.safe_child(node, 3)

		return ASM(
				type=TokenTypes.ASM,
				text=node.text.decode(),
				node=NODE(node, self.path),
				args=_cast(list[Cast], list(self.tokenize(args))) if args else [],
				out=_cast(list[Cast], list(self.tokenize(out))) if out else [],
				body=body.text.decode() if body.text is not None else ''
		)

	def _tokenize(self, node: tree_sitter.Node) -> Token:
		if node.type == 'word':
			return Word(TokenTypes.WORD, node.text.decode() if node.text else '', NODE(node, self.path), None)
		elif node.type == 'string':
			return Token(TokenTypes.STRING,
				unescape_string(node.text.decode()[1:-1]) if node.text else '', NODE(node, self.path))
		elif node.type == 'char':
			return Token(TokenTypes.CHAR,
				unescape_string(node.text.decode()[1:-1]) if node.text else '', NODE(node, self.path))
		elif node.type == 'number':
			return Token(TokenTypes.NUMBER, node.text.decode() if node.text else '', NODE(node, self.path))
		elif node.type == 'booleans':
			return Token(TokenTypes.BOOLEANS, node.text.decode() if node.text else '', NODE(node, self.path))
		elif node.type == 'operand':
			return self.parse_operand(node)
		elif node.type == 'builtin':
			return self.parse_builtin(node)
		elif node.type == 'identifier':
			return Token(TokenTypes.IDENTIFIER, node.text.decode() if node.text else '', NODE(node, self.path))
		elif node.type == 'sizeof':
			return self.parse_sizeof(node)
		elif node.type == 'definition':
			return self.parse_definition(node)
		elif node.type == 'include':
			return self.parse_include(node)
		elif node.type == 'cast':
			return self.parse_cast(node)
		elif node.type == 'pair':
			return self.parse_pair(node)
		elif node.type == 'accessor':
			return self.parse_accessor(node)
		elif node.type == 'in':
			raise Ignore()
		elif node.type == 'out':
			raise Ignore()
		elif node.type == 'comment':
			raise Ignore()
		elif node.type == 'asm':
			return self.parse_asm(node)
		raise NotImplementedError(node, node.type)
