CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror

ROOT := ../..
BUILD := $(ROOT)/target/release
PROGRAM := $(BUILD)/c-host-example

.PHONY: all run clean

all: $(PROGRAM)

$(BUILD)/libswxbx.so:
	cargo build --manifest-path $(ROOT)/Cargo.toml --release

$(PROGRAM): c-host.c $(ROOT)/include/swxbx.h $(BUILD)/libswxbx.so
	$(CC) $(CFLAGS) -I$(ROOT)/include $< -L$(BUILD) -lswxbx -o $@

run: $(PROGRAM)
	LD_LIBRARY_PATH=$(BUILD) $(PROGRAM) $(CURDIR)/census.txt $(CURDIR)/shapes.txt $(CURDIR)/frame.ppm

clean:
	rm -f $(PROGRAM) census.txt shapes.txt frame.ppm
