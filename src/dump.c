#include <unistd.h>
#include <stdint.h>

static void	putnbr(uint64_t x)
{
	if (x >= 10)
		putnbr(x / 10);
	char c = '0' + x % 10;
	write(1, &c, 1);
}

extern void	__dump(uint64_t x)
{
	putnbr(x);
	write(1, "\n", 1);
}

