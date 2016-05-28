SHELL = /bin/bash
HOME := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

FPC = f2py
FPCFLAGS = -lblas -llapack --fcompiler=gfortran

SRCDIR   := $(HOME)
SRCFILES = $(shell find $(SRCDIR) -name "*.f90")
OBJFILES = $(addsuffix .so, $(basename $(notdir $(SRCFILES))))

.PHONY : all clean
default_target: all

all: $(OBJFILES)
	@echo 'compile completed.'

clean:
	rm -rf $(HOME)/*.so
	rm -rf $(HOME)/*.so.dSYM

%.so: %.f90
	$(FPC) -c -m $(basename $(notdir $<)) $< $(FPCFLAGS)
