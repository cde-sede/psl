class LangExceptions(Exception):
	pass

class UnknownToken(LangExceptions): pass
class InvalidSyntax(LangExceptions): pass
class SymbolRedefined(LangExceptions): pass
class FileError(LangExceptions): pass
class Reporting(LangExceptions): pass

class Stopped(Exception): pass

class TypeCheckerException(LangExceptions): pass
class NotEnoughTokens(TypeCheckerException): pass
class InvalidType(TypeCheckerException): pass

class BlockException(TypeCheckerException): pass
class IfException(BlockException): pass
class ElifException(BlockException): pass
class ElseException(BlockException): pass
class WhileException(BlockException): pass

class MissingToken(BlockException): pass
class AddedToken(BlockException): pass

class TypeWarning(TypeCheckerException): pass
class StackNotEmpty(TypeWarning): pass
