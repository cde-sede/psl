from dataclasses import dataclass
from typing import cast as typing_cast
from itertools import batched, pairwise

from .classes import (
	Type,
	Types,
	TokenTypes,
	CONSTANTS,
	Builtins,
	Word,
	Syscalls,
	Operands,
	Token,
	Operand,
	Builtin,
	Syscall,
	Include,
	Cast,
	Sizeof,
	Accessor,
	Definition,
	Memory,
	Macro,
	ASM,
	Proc,
	Struct,
	With,
	Let,
	FlowControl,
	While,
	If,
	Elif,
	Else,
	StackType,
	Procs,

	Errors,
)

__all__ = ['TypeChecker']

def INVALIDTYPE(s: StackType, expected: Type):
	return Errors.InvalidType([s.origin[-1]], f"Invalid type: expected {expected} but received {s.typ}", None)

def INVALIDTYPE_L(s: StackType, expected: list[Type]):
	return Errors.InvalidType([s.origin[-1]], f"Invalid type: expected {' or '.join(str(i) for i in expected)} but received {s.typ}", None)

class TypeChecker:
	def __init__(self, *, stack=None, structs=None, locs=None, globs=None, macros=None, procs=None):
		self.stack: list[StackType] = stack if stack else []
		self.structs: list[Struct] = structs if structs else []
		self.locals: list[dict[str, StackType]] = locs if locs else []
		self.globals: list[Memory] = globs if globs else []
		self.macros: list[Macro] = macros if macros else []
		self.procs: list[Procs] = procs if procs else []

	def is_reserved(self, name: str) -> int:
		r = 0b00000
		if name in [ "syscall", "syscall1",
					"syscall2", "syscall3",
					"syscall4", "syscall5",
					"syscall6", "rsyscall",
					"rsyscall1", "rsyscall2",
					"rsyscall3", "rsyscall4",
					"rsyscall5", "rsyscall6",
					"dump", "udump", "cdump", "hexdump",
					"argc", "argv",
					"while", "do", "end",
					"if", "elif", "else",
					"proc", "memory", "macro",
					"include",
					"drop", "dup", "dup2", "swap",
					"over", "rot", "rrot", "exit",
					"true", "false", ]:
			r |= 0b00001
		if self.get_local(name) is not None:
			r |= 0b00010
		if self.get_macro(name) is not None:
			r |= 0b00100
		if self.get_proc(name) is not None:
			r |= 0b01000
		if self.get_struct(name) is not None:
			r |= 0b10000
		return r

	def get_struct(self, name: str) -> Struct | None:
		for i in self.structs:
			if i.name == name:
				return i
		return None

	def get_struct_bytype(self, t: Type) -> Struct | None:
		for i in self.structs:
			if i.typ == t:
				return i
		return None

	def get_local(self, name: str) -> StackType | None:
		for l in self.locals[::-1]:
			if l.get(name):
				return l[name]
		return None

	def get_global(self, name: str) -> Memory | None:
		for i in self.globals[::-1]:
			if i.name == name:
				return i
		return None

	def get_macro(self, name: str) -> Macro | None:
		for i in self.macros:
			if i.name == name:
				return i
		return None

	def get_proc(self, name: str) -> Procs | None:
		for i in self.procs:
			if i.name == name:
				return i
		return None

	def get_sign_proc(self, name: str, sign: list[Type]) -> Proc | None:
		for i in self.procs:
			if i.name == name:
				return i.by_signature(sign)
		return None

	def check_proc(self, t: Proc):
		l = {}
		for pair in t.args:
			l[pair.name.text] = StackType([pair.name], pair.cast.cast_type)
		tc = TypeChecker(
				macros=self.macros.copy(),
				procs=self.procs.copy(),
				structs=self.structs.copy(),
				locs=[l],
				globs=self.globals.copy(),
		)
		try:
			for i in t.body:
				tc.check(i)
		except Errors.TypeCheckerException as e:
			raise Errors.ProcedureError([t], "Error inside procedure body", e)
		if len(tc.stack) > len(t.out):
			for i in t.out:
				tc.stack.pop()
			raise Errors.UnhandledStackProcedure([i.origin[-1] for i in tc.stack], "Remaining data on stack")
		if len(tc.stack) < len(t.out):
			for i in tc.stack:
				t.out.pop()
			raise Errors.ProcedureReturn(t.out, "Missing return value")
		for top, out in zip(tc.stack[::-1], t.out):
			assert isinstance(out, Cast), '???'
			if top.typ != out.cast_type:
				raise INVALIDTYPE(top, out.cast_type)

	def check_length(self, n, token: Token):
		if len(self.stack) < n:
			raise Errors.NotEnoughArguments([token], f"Not enough arguments: expected {n} arguments but only {len(self.stack)} available")

	def check(self, token: Token):
		match token:
			case Token(type=TokenTypes.NUMBER) as t:
				self.stack.append(StackType([t], Types.INT))

			case Token(type=TokenTypes.STRING) as t:
				self.stack.append(StackType([t], Types.UINT))
				self.stack.append(StackType([t], Types.PTR[Types.CHAR]))

			case Token(type=TokenTypes.CHAR) as t:
				self.stack.append(StackType([t], Types.CHAR))

			case Token(type=TokenTypes.BOOLEANS) as t:
				self.stack.append(StackType([t], Types.BOOL))

			case Builtin(keyword=Builtins.DUP) as t:
				self.check_length(1, t)
				a = self.stack.pop()
				self.stack.append(StackType(a.origin+[t], a.typ))
				self.stack.append(StackType(a.origin+[t], a.typ))

			case Builtin(keyword=Builtins.DUP2) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()
				self.stack.append(StackType(b.origin+[t], b.typ))
				self.stack.append(StackType(a.origin+[t], a.typ))
				self.stack.append(StackType(b.origin+[t], b.typ))
				self.stack.append(StackType(a.origin+[t], a.typ))

			case Builtin(keyword=Builtins.SWAP) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()
				self.stack.append(StackType(a.origin+[t], a.typ))
				self.stack.append(StackType(b.origin+[t], b.typ))

			case Builtin(keyword=Builtins.OVER) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()
				self.stack.append(StackType(b.origin+[t], b.typ))
				self.stack.append(StackType(a.origin+[t], a.typ))
				self.stack.append(StackType(b.origin+[t], b.typ))

			case Builtin(keyword=Builtins.ROT) as t:
				self.check_length(3, t)
				a = self.stack.pop()
				b = self.stack.pop()
				c = self.stack.pop()
				self.stack.append(StackType(b.origin+[t], a.typ))
				self.stack.append(StackType(a.origin+[t], b.typ))
				self.stack.append(StackType(c.origin+[t], b.typ))

			case Builtin(keyword=Builtins.RROT) as t:
				self.check_length(3, t)
				a = self.stack.pop()
				b = self.stack.pop()
				c = self.stack.pop()
				self.stack.append(StackType(a.origin+[t], b.typ))
				self.stack.append(StackType(c.origin+[t], b.typ))
				self.stack.append(StackType(b.origin+[t], a.typ))

			case Builtin(keyword=Builtins.EXIT) as t:
				self.check_length(1, t)
				a = self.stack.pop()

			case Builtin(keyword=Builtins.DROP) as t:
				self.check_length(1, t)
				on_stack = self.stack.pop()

			case Builtin(keyword=Builtins.DUMP) as t:
				self.check_length(1, t)
				on_stack = self.stack.pop()

			case Builtin(keyword=Builtins.UDUMP) as t:
				self.check_length(1, t)
				on_stack = self.stack.pop()

			case Builtin(keyword=Builtins.CDUMP) as t:
				self.check_length(1, t)
				on_stack = self.stack.pop()
				if on_stack.typ != Types.CHAR:
					raise INVALIDTYPE(on_stack, Types.CHAR)

			case Builtin(keyword=Builtins.HEXDUMP) as t:
				self.check_length(1, t)
				on_stack = self.stack.pop()

			case Builtin(keyword=Builtins.ARGV) as t:
				self.stack.append(StackType([t], Types.PTR[Types.PTR[Types.CHAR]]))

			case Builtin(keyword=Builtins.ARGC) as t:
				self.stack.append(StackType([t], Types.INT))

			case Syscall() as t:
				self.check_length(t.nargs + 1, t)
				for i in range(t.nargs + 1):
					on_stack = self.stack.pop()
				self.stack.append(StackType([t], Types.INT))

			case Operand(operand=Operands.PLUS) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if -b.typ == Types.PTR:
					if a.typ not in [Types.SHORT, Types.USHORT,
									 Types.INT, Types.UINT,
									 Types.LONG, Types.ULONG]:
						raise INVALIDTYPE_L(a, [Types.SHORT, Types.USHORT,
									 Types.INT, Types.UINT,
									 Types.LONG, Types.ULONG])
					self.stack.append(StackType([t], b.typ))
				elif -a.typ == Types.PTR:
					if b.typ not in [Types.SHORT, Types.USHORT,
									 Types.INT, Types.UINT,
									 Types.LONG, Types.ULONG]:
						raise INVALIDTYPE_L(a, [Types.SHORT, Types.USHORT,
									 Types.INT, Types.UINT,
									 Types.LONG, Types.ULONG])
					self.stack.append(StackType([t], a.typ))
				elif b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.MINUS) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if -b.typ == Types.PTR:
					if a.typ not in [Types.SHORT, Types.USHORT,
									 Types.INT, Types.UINT,
									 Types.LONG, Types.ULONG]:
						raise INVALIDTYPE_L(a, [Types.SHORT, Types.USHORT,
									 Types.INT, Types.UINT,
									 Types.LONG, Types.ULONG])
					self.stack.append(StackType([t], b.typ))
				elif -a.typ == Types.PTR:
					raise Errors.InvalidType([a.origin[-1]], "Invalid type: can't be a ptr")
				elif b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.MUL) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if -a.typ == Types.PTR: raise Errors.InvalidType([a.origin[-1]], "Invalid type: can't be a ptr")
				elif -b.typ == Types.PTR: raise Errors.InvalidType([b.origin[-1]], "Invalid type: can't be a ptr")
				elif b.typ @ a.typ:
					if b.typ in [Types.LONG, Types.ULONG, Types.QWORD]:
						self.stack.append(StackType([t], b.typ)) # higher part
						self.stack.append(StackType([t], Types.ULONG)) # lower part
					else:
						promotion_order = [Types.CHAR, Types.SHORT, Types.INT, Types.LONG,
							Types.UCHAR, Types.USHORT, Types.UINT, Types.ULONG,
							Types.BYTE, Types.WORD, Types.DWORD, Types.QWORD
						]

						nt = promotion_order[promotion_order.index(b.typ) + 1]
						self.stack.append(StackType([t], nt))
						t.size = b.typ.size
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.DIVMOD) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if -a.typ == Types.PTR: raise Errors.InvalidType([a.origin[-1]], "Invalid type: can't be a ptr")
				elif -b.typ == Types.PTR: raise Errors.InvalidType([b.origin[-1]], "Invalid type: can't be a ptr")
				elif b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ)) # /
					self.stack.append(StackType([t], b.typ)) # %
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.DIV) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if -a.typ == Types.PTR: raise Errors.InvalidType([a.origin[-1]], "Invalid type: can't be a ptr")
				elif -b.typ == Types.PTR: raise Errors.InvalidType([b.origin[-1]], "Invalid type: can't be a ptr")
				elif b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ)) # /
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.MOD) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if -a.typ == Types.PTR: raise Errors.InvalidType([a.origin[-1]], "Invalid type: can't be a ptr")
				elif -b.typ == Types.PTR: raise Errors.InvalidType([b.origin[-1]], "Invalid type: can't be a ptr")
				elif b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ)) # %
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.INC) as t:
				self.check_length(1, t)
				a = self.stack.pop()

				if not a.typ.is_builtin:
					raise Errors.InvalidType([a.origin[-1]], "Invalid type: must be a built-in type (such as int, char, ptr etc)")
				self.stack.append(StackType(a.origin + [t], a.typ))

			case Operand(operand=Operands.DEC) as t:
				self.check_length(1, t)
				a = self.stack.pop()

				if not a.typ.is_builtin:
					raise Errors.InvalidType([a.origin[-1]], "Invalid type: must be a built-in type (such as int, char, ptr etc)")
				self.stack.append(StackType(a.origin + [t], a.typ))

			case Operand(operand=Operands.BRSH) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if not b.typ.is_builtin:
					raise Errors.InvalidType([b.origin[-1]], "Invalid type: must be a built-in type (such as int, char, ptr etc)")
				if a.typ @ Types.BYTE:
					self.stack.append(StackType([t], b.typ))
				else:
					raise Errors.InvalidType([b.origin[-1]], "Invalid type: can only bitshift with byte or byte equivalent")

			case Operand(operand=Operands.BLSH) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if not b.typ.is_builtin:
					raise Errors.InvalidType([b.origin[-1]], "Invalid type: must be a built-in type (such as int, char, ptr etc)")
				if a.typ @ Types.BYTE:
					self.stack.append(StackType([t], b.typ))
				else:
					raise Errors.InvalidType([b.origin[-1]], "Invalid type: can only bitshift with byte or byte equivalent")

			case Operand(operand=Operands.BAND) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ))
				else:
					raise Errors.InvalidType([a, b], "Invalid type: both types must be of the same size")

			case Operand(operand=Operands.BOR) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ))
				else:
					raise Errors.InvalidType([a, b], "Invalid type: both types must be of the same size")

			case Operand(operand=Operands.BXOR) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ @ a.typ:
					self.stack.append(StackType([t], b.typ))
				else:
					raise Errors.InvalidType([a, b], "Invalid type: both types must be of the same size")

			case Operand(operand=Operands.EQ) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ == a.typ:
					self.stack.append(StackType([t], Types.BOOL))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.NE) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ == a.typ:
					self.stack.append(StackType([t], Types.BOOL))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.GT) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ == a.typ:
					self.stack.append(StackType([t], Types.BOOL))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.GE) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ == a.typ:
					self.stack.append(StackType([t], Types.BOOL))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.LT) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ == a.typ:
					self.stack.append(StackType([t], Types.BOOL))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Operand(operand=Operands.LE) as t:
				self.check_length(2, t)
				a = self.stack.pop()
				b = self.stack.pop()

				if b.typ == a.typ:
					self.stack.append(StackType([t], Types.BOOL))
				else:
					raise INVALIDTYPE(a, b.typ)

			case Struct() as t:
				if r := self.is_reserved(t.name):
					if r & 0b1:
						raise Errors.ReservedKeyword(t, "is a reserved keyword")
					if r:
						raise Errors.Redefinition(t, "is taken")
				self.structs.append(t)
				for group in t.methods.values():
					for proc in group.procs:
						self.check_proc(proc)

			case Sizeof() as t:
				self.stack.append(StackType([t], Types.INT))

			case Word() as t:
				if v := self.get_local(t.text):
					self.stack.append(StackType(v.origin + [t], v.typ))
				elif v := self.get_global(t.text):
					self.stack.append(StackType([t, v], Types.PTR[v.typ]))
				elif v := self.get_macro(t.text):
					for i in v.body:
						self.check(i)
				elif v := self.get_proc(t.text):
					f = None
					if not self.stack:
						f = v.by_signature([])
					for n in reversed(range(len(self.stack)+1)):
						f = v.by_signature([i.typ for i in self.stack[::-1][:n]])
						if f:
							break
					if not f:
						if len(v.procs) > 1:
							
							
							raise Errors.NoOverloadMatch([t], f"No matching function overload\n\t{'\n\t'.join(', '.join(f"{a.typ} {'==' if a == b else '!='} {b.cast_type}" for a, b in zip(self.stack[::-1], i.signature())) for i in v.procs)}")

						raise Errors.InvalidType([t], f"Invalid type: procedure expects [{', '.join(str(j.cast_type) for j in v.procs[0].signature())}] "
									   f"but received [{', '.join(str(a.typ) for a, b in zip(self.stack[::-1], v.procs[0].signature()))}]")
					t.data = f
					for i in f.signature():
						self.stack.pop()
					for i in f.out[::-1]:
						self.stack.append(StackType([t, i], i.cast_type))
				else:
					raise Errors.UnknownWord([t], "Unknown word")

			case Accessor(var=Type(is_builtin=True), field='') as t:
				if t.typ:	# set
					self.check_length(2, t)
					addr = self.stack.pop()
					value = self.stack.pop()
					if addr.typ != Types.PTR[t.var]:
						raise INVALIDTYPE(addr, Types.PTR[t.var])
					if value.typ != t.var:
						raise INVALIDTYPE(value, t.var)
				else:		# get
					self.check_length(1, t)
					addr = self.stack.pop()
					if addr.typ != Types.PTR[t.var]:
						raise INVALIDTYPE(addr, Types.PTR[t.var])
					self.stack.append(StackType([t], t.var))

			case Accessor(var=Type(is_builtin=True)) as t:
				raise NotImplementedError("REPORTING", t)

			case Accessor(var=Type(is_builtin=False), field='') as t:
				struct = self.get_struct_bytype(t.var)
				if struct is None:
					raise Errors.UnknownStructure([t], f"No structure called {t.var.name}")

				if t.typ:
					procs = struct.constructor()
					if procs is None:
						raise Errors.NoConstructor([t], f"No constructor for structure {struct.name}")
				else:
					procs = struct.getter()
					if procs is None:
						raise Errors.NoGetter([t], f"No getter for structure {struct.name}")

				f = None
				if not self.stack:
					f = procs.by_signature([])
				for n in range(len(self.stack)):
					f = procs.by_signature([i.typ for i in self.stack[::-1][:n+1]])
					if f:
						break
				if not f:
					if len(procs.procs) > 1:
						raise Errors.NoOverloadMatch([t], "No matching function overload")
					raise Errors.InvalidType([t], f"Invalid type: procedure expects [{', '.join(str(j.cast_type) for j in procs.procs[0].signature())}] "
								  f"but received [{', '.join(str(a.typ) for a, b in zip(self.stack[::-1], procs.procs[0].signature()))}]"
							  )
				t.data = f

				if t.typ:	# set
					self.check_length(len(f.signature()), t)
					for i in f.signature():
						self.stack.pop()
					for i in f.out[::-1]:
						self.stack.append(StackType([t, i], i.cast_type))
				else:		# get
					self.check_length(len(f.signature()), t)
					for i in f.signature():
						self.stack.pop()
					for i in f.out[::-1]:
						self.stack.append(StackType([t, i], i.cast_type))

			case Accessor(var=Type(is_builtin=False)) as t:
				struct = self.get_struct_bytype(t.var)
				if struct is None:
					raise Errors.UnknownStructure([t], f"No structure called {t.var.name}")
				field = struct.get_field(t.field)
				if field is None:
					raise Errors.UnknownField([t], f"No field {t.field} for {struct.name}")
				t.data = field
				if t.typ:	# set
					self.check_length(2, t)
					addr = self.stack.pop()
					value = self.stack.pop()
					if addr.typ != Types.PTR[t.var]:
						raise INVALIDTYPE(addr, Types.PTR[t.var])
					if value.typ != field.pair.cast.cast_type:
						raise INVALIDTYPE(value, field.pair.cast.cast_type)
				else:		# get
					self.check_length(1, t)
					addr = self.stack.pop()
					if addr.typ != Types.PTR[t.var]:
						raise INVALIDTYPE(addr, Types.PTR[t.var])
					self.stack.append(StackType([t], field.pair.cast.cast_type))

			case Cast() as t:
				self.check_length(1, t)
				on_stack = self.stack.pop()
				self.stack.append(StackType(on_stack.origin + [t], t.cast_type))

			case Include() as t:
				tc = TypeChecker()
				for i in t.body:
					tc.check(i)
				self.macros += tc.macros
				self.procs += tc.procs
				self.structs += tc.structs

			case Macro() as t:
				if r := self.is_reserved(t.name):
					if r & 0b1:
						raise Errors.ReservedKeyword(t, "is a reserved keyword")
					if r:
						raise Errors.Redefinition(t, "is taken")
				self.macros.append(t)

			case ASM() as t:
				self.check_length(len(t.args), t)
				for a, b in zip(t.args, self.stack[::-1]):
					if a.cast_type != b.typ:
						raise Errors.InvalidType([t, a], f"Inline ASM expecting {a.cast_type} but received {b.typ}")
				for i in t.args:
					self.stack.pop()
				for i in t.out[::-1]:
					self.stack.append(StackType([t, i], i.cast_type))

			case Proc() as t:
				r = self.is_reserved(t.name)
				if r & 0b1:
					raise Errors.ReservedKeyword(t, "is a reserved keyword")
				if r & 0b0111:
					raise Errors.Redefinition(t, "is taken")
				if r & 0b1000:
					p = typing_cast(Procs, self.get_proc(t.name))
					if p.by_signature(list(map(lambda x: x.cast_type, t.signature()))):
						raise Errors.Redefinition(t, "a procedure with the same signature has already been defined")
					p.procs.append(t)
				else:
					self.procs.append(Procs(t.name, [t]))
				self.check_proc(t)

			case If() as t:
				def runcopy(l: list[Token]):
					tc = TypeChecker(
						stack=self.stack.copy(),
						macros=self.macros.copy(),
						procs=self.procs.copy(),
						structs=self.structs.copy(),
						locs=self.locals.copy(),
						globs=self.globals.copy(),
					)
					for i in l:
						tc.check(i)
					return tc

				def cmpstack(a, b):
					if len(a) > len(b):
						raise Errors.IfError([i.origin[-1] for i in a[:-len(b)]], "If must not alter stack")
					if len(a) < len(b):
						raise Errors.IfError([i.origin[-1] for i in b[:-len(a)]], "If must not alter stack")
					for x, y in zip(a, b):
						if x.typ != y.typ:
							raise INVALIDTYPE(x, y.typ)

				cond = runcopy(t.condition)
				if not (len(cond.stack) == len(self.stack) + 1 and cond.stack[-1].typ == Types.BOOL):
					raise Errors.IfError([cond.stack[-1].origin[-1]], "If must not alter stack, except for a singular bool required for the do")

				body = runcopy(t.body)

				els = None
				if t.else_:
					els = runcopy(t.else_.body)

				elifs = []
				for i in t.elifs:
					ec = runcopy(i.condition)
					if not (len(ec.stack) == len(self.stack) + 1 and ec.stack[-1].typ == Types.BOOL):
						raise Errors.IfError([cond.stack[-1].origin[-1]], "If must not alter stack, except for a singular bool required for the do")
					elifs.append(runcopy(i.body))

				if els:
					for a, b in pairwise([body, *elifs, els]):
						cmpstack(a.stack, b.stack)
					self.stack = els.stack
				else:
					for a, b in pairwise([self, body, *elifs]):
						cmpstack(a.stack, b.stack)

			case While() as t:
				def runcopy(l: list[Token]):
					tc = TypeChecker(
						stack=self.stack.copy(),
						macros=self.macros.copy(),
						procs=self.procs.copy(),
						structs=self.structs.copy(),
						locs=self.locals.copy(),
						globs=self.globals.copy(),
					)
					for i in l:
						tc.check(i)
					return tc

				def cmpstack(a, b):
					if len(a) > len(b):
						raise Errors.WhileError([i.origin[-1] for i in a[:-len(b)]], "While must not alter stack")
					if len(a) < len(b):
						raise Errors.WhileError([i.origin[-1] for i in b[:-len(a)]], "While must not alter stack")
					for x, y in zip(a, b):
						if x.typ != y.typ:
							raise INVALIDTYPE(x, y.typ)
				
				cond = runcopy(t.condition)
				if not (len(cond.stack) == len(self.stack) + 1 and cond.stack[-1].typ == Types.BOOL):
					raise Errors.WhileError([cond.stack[-1].origin[-1]], "While must not alter stack, except for a singular bool required for the do")
				body = runcopy(t.body)
				cmpstack(body.stack, self.stack)

			case Memory() as t:
				if self.is_reserved(t.name):
					raise NotImplementedError("REPORTING")
				tc = TypeChecker(
					macros=self.macros.copy(),
					globs=self.globals.copy(),
				)
				for i in t.body:
					tc.check(i)
				if len(tc.stack) != 1:
					raise NotImplementedError("REPORTING")
				
				self.globals.append(t)

			case With() as t:
				self.check_length(len(t.variables), t)
				l = {}
				for pair in t.variables:
					top = self.stack.pop()
					if pair.cast.cast_type != top.typ:
						raise INVALIDTYPE(top, pair.cast.cast_type)
					l[pair.name.text] = StackType(top.origin + [pair.name], pair.cast.cast_type)
				self.locals.append(l)
				for i in t.body:
					self.check(i)
				self.locals.pop()

			case Let() as t:
				self.check_length(len(t.variables), t)
				l = {}
				for pair in t.variables:
					top = self.stack.pop()
					l[pair.name.text] = StackType([pair.name], Types.PTR[pair.cast.cast_type])
				self.locals.append(l)
				for i in t.body:
					self.check(i)
				self.locals.pop()
			case _:
				raise NotImplementedError(str(token))
