#include <stdlib.h>
#include <stdio.h>

extern void	*__malloc(size_t n)
{
	return malloc(n);
}

extern void	__free(void *ptr)
{
	free(ptr);
}
