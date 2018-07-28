DIR=bin
NAME=fractal-gui

all: build

build: $(DIR)/$(NAME)

$(DIR)/$(NAME): $(DIR)/Makefile
	@make -C $(DIR) all -j10 --no-print-directory

$(DIR)/Makefile: CMakeLists.txt $(DIR)
	@cmake $(DIR)

$(DIR):
	@mkdir -p $(DIR)

clean:
	@make -C $(DIR) clean -j10 --no-print-directory

reset:
	@rm -rf $(DIR)

run:
	@$(DIR)/$(NAME)
