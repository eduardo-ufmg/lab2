# Makefile for building imc.c and pi.c (NI-DAQmx + GSL)
#
# Usage:
#   make                # build both imc and pi
#   make clean          # remove build artifacts
#
# Override any of these variables on the command line, e.g.:
#   make CC=clang CFLAGS="-O2 -g" NIDAQMX_LIB=/opt/natinst/lib

# Compiler
CC ?= gcc

# Default compiler flags
CFLAGS ?= -Wall -Wextra

# Try to pick up GSL flags via pkg-config when available
GSL_CFLAGS ?= $(shell pkg-config --cflags gsl 2>/dev/null)
GSL_LIBS ?= $(shell pkg-config --libs gsl 2>/dev/null)

# NI-DAQmx include/lib paths (adjust if your NI install is in a non-standard location)
# Example: make NIDAQMX_LIB=/opt/natinst/lib NIDAQMX_INCLUDE=/opt/natinst/include
NIDAQMX_INCLUDE ?= /usr/include
NIDAQMX_LIB ?= /usr/lib/x86_64-linux-gnu

# Link in the NI-DAQmx library
NIDAQMX_LIBS ?= -L$(NIDAQMX_LIB) -lnidaqmx

# Final flags
CPPFLAGS += -I$(NIDAQMX_INCLUDE) $(GSL_CFLAGS)
LDFLAGS += -L$(NIDAQMX_LIB)
LDLIBS += -lnidaqmx -lpthread -lm $(GSL_LIBS)

# Targets
.PHONY: all clean info
all: imc pi

imc: imc.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

pi: pi.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -f imc pi imc.o pi.o

info:
	@echo "CC = $(CC)"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "CPPFLAGS = $(CPPFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"
	@echo "LDLIBS = $(LDLIBS)"
	@echo "NIDAQMX_INCLUDE = $(NIDAQMX_INCLUDE)"
	@echo "NIDAQMX_LIB = $(NIDAQMX_LIB)"
	@echo "GSL_CFLAGS = $(GSL_CFLAGS)"
	@echo "GSL_LIBS = $(GSL_LIBS)"
