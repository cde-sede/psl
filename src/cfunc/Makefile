OBJ_DIR = objs

SRCS = dump.c \
	   memory.c

OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SRCS))

CC = gcc
CFLAGS = -lasm -g -ggdb

all: $(OBJS)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re
