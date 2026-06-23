VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PYTHON ?= $(VENV_PYTHON)
SYSTEM_PYTHON ?= python3
PIP := $(PYTHON) -m pip
DEPS_STAMP := $(VENV)/.deps-stamp
ENGINE ?= taichi
ARCH ?= auto
CONFIG ?=
EXTRA_ARGS ?=
OUT ?= out/latest
FPS ?= 30
FRAMES ?= 240
GRID ?= 72
WIDTH ?= 540
HEIGHT ?= 540
RAY_STEPS ?= 104
SUBSTEPS ?= 2
USE_CONFIG := $(if $(and $(strip $(CONFIG)),$(filter taichi,$(ENGINE))),1,)

USER_VAR_ORIGINS := command line environment environment override

define from_user
$(filter $(USER_VAR_ORIGINS),$(origin $(1)))
endef

define maybe_arg
$(if $(call from_user,$(1)),--$(2) $($(1)),)
endef

ifeq ($(ENGINE),taichi)
RUNNER := main_taichi.py
ifeq ($(USE_CONFIG),)
ENGINE_ARGS := --arch $(ARCH)
else
ENGINE_ARGS := $(call maybe_arg,ARCH,arch)
endif
else ifeq ($(ENGINE),numpy)
RUNNER := main.py
ENGINE_ARGS :=
else
$(error ENGINE must be 'taichi' or 'numpy')
endif
RUN_PREFIX := $(PYTHON) $(RUNNER) $(ENGINE_ARGS)
CONFIG_ARG := $(if $(USE_CONFIG),--config $(CONFIG),)

ifneq ($(USE_CONFIG),)
RENDER_BASE_ARGS := \
	$(CONFIG_ARG) \
	$(call maybe_arg,OUT,out) \
	$(call maybe_arg,FRAMES,frames) \
	$(call maybe_arg,FPS,fps) \
	$(call maybe_arg,GRID,grid) \
	$(call maybe_arg,WIDTH,width) \
	$(call maybe_arg,HEIGHT,height) \
	$(call maybe_arg,RAY_STEPS,ray-steps) \
	$(call maybe_arg,SUBSTEPS,substeps)
ENCODE_BASE_ARGS := \
	$(CONFIG_ARG) \
	$(call maybe_arg,OUT,out) \
	$(call maybe_arg,FPS,fps)
else
RENDER_BASE_ARGS := \
	--out $(OUT) \
	--frames $(FRAMES) \
	--fps $(FPS) \
	--grid $(GRID) \
	--width $(WIDTH) \
	--height $(HEIGHT) \
	--ray-steps $(RAY_STEPS) \
	--substeps $(SUBSTEPS)
ENCODE_BASE_ARGS := \
	--out $(OUT) \
	--fps $(FPS)
endif

.PHONY: venv deps frames mp4 gif all update clean

$(VENV_PYTHON):
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

$(DEPS_STAMP): requirements.txt | $(VENV_PYTHON)
	$(PIP) install -r requirements.txt
	touch $(DEPS_STAMP)

venv: $(VENV_PYTHON)

deps: $(DEPS_STAMP)

frames: $(DEPS_STAMP)
	$(RUN_PREFIX) \
		$(RENDER_BASE_ARGS) \
		--overwrite $(EXTRA_ARGS)

mp4: frames
	$(RUN_PREFIX) \
		$(ENCODE_BASE_ARGS) \
		--encode-only \
		--mp4 \
		--overwrite $(EXTRA_ARGS)

gif: frames
	$(RUN_PREFIX) \
		$(ENCODE_BASE_ARGS) \
		--encode-only \
		--gif \
		--overwrite $(EXTRA_ARGS)

all:
	$(RUN_PREFIX) \
		$(RENDER_BASE_ARGS) \
		--mp4 \
		--gif \
		--overwrite $(EXTRA_ARGS)

update: all

clean:
	rm -rf out $(VENV)
