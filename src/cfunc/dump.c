#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

extern void	__hexdump(uint64_t x)
{
	printf("0x%lx\n", x);
}

extern void	__udump(uint64_t x)
{
	printf("%lu\n", x);
}

extern void	__dump(uint64_t x)
{
	printf("%ld\n", x);
}

extern void	__cdump(char c)
{
	write(1, &c, 1);
//	printf("%c\n", c);
}


extern void	__printline(const unsigned char *ptr)
{
	printf("%s\n", ptr);
}
