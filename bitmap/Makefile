DIR=bin
NAME=bitmap_test

all: build

build: $(DIR)/$(NAME)

$(DIR)/$(NAME): $(DIR)/Makefile
	@make -C $(DIR) all -j10 --no-print-directory

$(DIR)/Makefile: CMakeLists.txt $(DIR)
	@cd $(DIR); cmake ..

$(DIR):
	@mkdir -p $(DIR)

clean:
	@make -C $(DIR) clean -j10 --no-print-directory

reset:
	@rm -rf $(DIR)

run: build
	@$(DIR)/$(NAME)
