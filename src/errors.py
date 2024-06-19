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

class IfException(TypeCheckerException): pass
class ElseException(TypeCheckerException): pass
class WhileException(TypeCheckerException): pass
class MissingToken(TypeCheckerException): pass
class AddedToken(TypeCheckerException): pass

class TypeWarning(TypeCheckerException): pass
class StackNotEmpty(TypeWarning): pass
