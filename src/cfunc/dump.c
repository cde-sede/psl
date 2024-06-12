#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

extern void	__hexdump(uint64_t x)
{
	printf("0x%lx\n", x);
	fflush(stdout);
}

extern void	__udump(uint64_t x)
{
	printf("%lu\n", x);
	fflush(stdout);
}

extern void	__dump(uint64_t x)
{
	printf("%ld\n", x);
	fflush(stdout);
}

extern void	__cdump(char c)
{
	write(1, &c, 1);
}
