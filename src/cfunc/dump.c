#include <stdint.h>
#include <stdio.h>

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

