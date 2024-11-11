# PSL
### A stack based compiled toy language written in python


## Example

```
include "string.psl"

proc main
    ac :int
    av :char**
in :int out
    "Hello world" putpstr
    0
end

argv argc main exit
```


## Features
### Parser

The parser for the compiler was made using tree-sitter to simplify syncing the syntax highlighting and the compiler parser/lexer.
It is most certainly not the best way to do so, and probably the worse use of tree-sitter, unless you have a good reason to do so,
I believe copying me to be a bad idea.  

To get more information on the parser, [see this](https://github.com/cde-sede/tree-sitter-psl)

### Compiler

This is the list of what the language supports currently. **(As the project is work in progress, everything may change constantly)**

#### Literals

##### Integer
A number is defined as an optional `-` followed by any number of digits.
```
1 -2 3333333333333
```

##### Booleans
Booleans is defined as either `true` or `false`
```
true false
```

##### Char
A char is defined as single quotes containing an optional `\` followed by any character. 
```
'a' 'b' '\n' '\t'
```

##### String
A string is defined as double quotes containing any escaped sequence or characters
```
"abc\n" "\033[31mRED\033[0m" "\"Escaped quotes\""
```

For the compiler, a string is two elements on the stack, in effect, two values are pushed on the stack, the size then the pointer to the first element.
```
"Hello, World!" 1 1 syscall3 drop
```
Is the PSL equivalent of the following C code
```C
write(1, "Hello, World!", 13);
```

##### Built-ins/Operators

PSL comes with functions and operators to manipulate the stack and integers

| Name | Signature | Description |
|------|-----------|-------------|
|swap|a b -> b a|swaps 2 elements on top of the stack|
|dup|a -> a a|duplicates the top element|
|dup2|a b -> a b a b|duplicates 2 elements on top of the stack|
|drop|a -> |pops and ignore the top element|
|dump|a -> |pops and prints the top element|
|udump|a -> |pops and prints the top element as an unsigned number|
|cdump|a :char -> |pops and prints the top element as a char|
|hexdump|a -> |pops and prints the top element as an hex number|
|rot|a b c -> b c a |rotates the top 3 elements|
|rrot|a b c -> c a b |rotates the top 3 elements in the other direction|
|+|a :t b :t -> (a + b) :t |adds two elements on the stack|
|-|a :t b :t -> (a - b) :t |substracts two elements on the stack|
|*|a :t b :t -> (a * b) :t |multiplies two elements on the stack|
|/%|a :t b :t -> (a / b) :t (a % b) :t |divides two elements on the stack and push the divisor and remainder|
|/|a :t b :t -> (a / b) :t |divides two elements on the stack and push the divisor|
|%|a :t b :t -> (a % b) :t |divides two elements on the stack and push the remainder|
|++|a :t -> (a + 1) :t |increments the top element|
|--|a :t -> (a - 1) :t |decrements the top element|
|<<|a :t b :t -> (a << b) :t |bitshift towards the left the top element|
|>>|a :t b :t -> (a >> b) :t |bitshift towards the right the top element|
|&|a :t b :t -> (a & b) :t |bitwise and two elements on the stack|
|\||a :t b :t -> (a \| b) :t |bitwise or two elements on the stack|
|^|a :t b :t -> (a ^ b) :t |bitwise xor two elements on the stack|
|==|a :t b :t -> (a == b) :bool |checks for equality of two elements on the stack|
|!=|a :t b :t -> (a != b) :bool |checks for inequality of two elements on the stack|
|>|a :t b :t -> (a > b) :bool |checks if the 2nd element on the stack is greated than the top|
|<|a :t b :t -> (a < b) :bool |checks if the 2nd element on the stack is smaller than the top|
|>=|a :t b :t -> (a >= b) :bool |checks if the 2nd element on the stack is greated or equal than the top|
|<=|a :t b :t -> (a <= b) :bool |checks if the 2nd element on the stack is smaller or equal than the top|

##### Casts

The language roughly checks whether or not the program is correct based on types and stack growth.
But, if necessary, you can cheat the type using a cast.
Casts are defined using a color `:` then the type you want to cast to
```
'a' 1 + dump // will raise an error
             //  Error:line 1
             //  'a' 1 + dump
             //  
             //      ^
             //  Invalid type: expected char but received int

'a' :int 1 + dump
// -> b

// OR

'a' 1 :char + dump
// -> b
```


##### Structures

Structures allow for user defined types.
They are defined using the `struct` keyword, followed by a name and a list of pairs of name and type.
Procedures with name `$` and `~` can be defined to interact with default writer and reader accessors (in this order).
For example the `str` type defined in the std lib `string.psl`

```
struct str
    count :int
    data :char*

    proc $
        this :str*
        arg :char*
        len :int
    in
        len this !str.count
        arg this !str.data
    end

    proc $
        this :str*
        arg :char*
        len :uint
    in
        len :int this !str.count
        arg this !str.data
    end

    proc ~
        this :str*
    in
        :char* 
        :uint 
    out
        this @str.count :uint
        this @str.data
    end
end
```

**Structures can only be used a pointers on the stack (for now)**


##### Accessors

Accessors allows to read or write memory.
The first character must be @ for read and ! for write, then the type of the data you want to write.
For reading, the accessor only expects the address as a pointer of the type
But for writing, you first need to push the values to be written
```
memory a :int 1 end

a @int dump // -> 0
100 a !int
a @int dump // -> 100
```

You can access a field of a structure using a dot, for example
```
include "string.psl"
memory s :str 1 end
"abcd" s !str.data :int s !str.count
s putstr // -> abcd

```

And, with default accessors
```
include "string.psl"
memory s :str 1 end
"abcd" s !str
s @str.count s @str.data 1 1 syscall3 drop  // -> abcd
s @str 1 1 syscall3 drop                    // -> abcd
```

##### Syscalls

Syscalls are made using `syscall<N>` where `<N>` is the number of arguments of the syscall.
Arguments are pushed in reverse order, ending with the syscall id.
The `rax` register resulting from the syscall is pushed on the stack.
For example write(2)
```C
ssize_t write(int fd, const void *buf, size_t count);
```
Will be called using
```
count buf fd 1 syscall3 drop
```

##### Arguments
Command line arguments are accessed using `argc` (:int) and `argv` (:char**)
`argv` returns a pointer to the first argument.

##### Control flow
PSL allows for basic flow control through `if` and `while`
They are bound under a simple rule, the state of the stack (size and types) must remain the same across all flow control cases.
Example:

```
// argc must stay on the stack
argc if dup 1 == do
    "There is one argument on the stack" 1 1 syscall3 drop
end drop

// without ther dup
argc if 1 == do     // Error:line 1
                    // argc if 1 == do
                    //           ^^
                    // If must not alter stack, except for a singular bool required for the do
    "There is one argument on the stack" 1 1 syscall3 drop
end

// allowed because an int is pushed no matter what
if true do
    1
else
    2
end

// The types are different (:int vs :char vs :bool)
if false do
    1               // Error:line 2
elif true do        //     1
    'a'             // 
else                //     ^
    false           // Invalid type: expected char but received int
end
```

```
0 while dup 10 < do
    dup dump
    ++
end

0 while dup 10 < do
    dump            // Error:line 1
end                 // 0 while dup 10 < do
                    // 
                    // ^
                    // While must not alter stack (got removed)


0 while dup 10 < do
    1               // Error:line 2
end                 //      1
                    //  
                    //      ^
                    //  While must not alter stack (got added)
```

The lack of `break` in while loops allows for some creative constructions.

```
proc strncpy
    dst :char*
    src :char*
    n :int
in
    0 while
        if dup n >= do
            false
        else
            if dup src + @char :bool do
                dup src + @char
                over dst + !char
                true
            else
                false
            end
        end
    do
        ++
    end drop
end

```

##### Procedures
 proc in out end

##### Memory allocations
 with let memory

##### Include
 include

##### Type checker

##### Inline assembly
 ( int out )
    
## References

- Linux syscalls: https://chromium.googlesource.com/chromiumos/docs/+/master/constants/syscalls.md
- x86-64 calling convention: https://en.wikipedia.org/wiki/X86_calling_conventions
- Tree sitter: https://tree-sitter.github.io/tree-sitter/

## TODO

- [ ] Hex/octal/bin literals
- [ ] Structures on stack
- [ ] Chained accessors (@a.b @b.c @c.d -> @a.b.c.d)
- [ ] Remove the dumb recursive compiler
- [ ] Fix local allocs in loops
- [ ] Main entrypoint
- [ ] Self hosting + bootstrap
- [ ] Optimizations

- [ ] Interpreter (again)
- [ ] Testing (again)
