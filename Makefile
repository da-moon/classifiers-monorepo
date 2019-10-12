# OS specific part
# -----------------
ifeq ($(OS),Windows_NT)
    CLEAR = cls
    LS = dir
    TOUCH =>>
    RM = del /F /Q
    CPF = copy /y
    RMDIR = -RMDIR /S /Q
    MKDIR = -mkdir
    DEVNUL := NUL
    WHICH := where
    CMDSEP = &
    ERRIGNORE = 2>NUL || (exit 0)
    SEP=\\
else
    CLEAR = clear
    LS = ls
    TOUCH = touch
    CPF = cp -f
    RM = rm -rf
    RMDIR = rm -rf
    CMDSEP = ;
    MKDIR = mkdir -p
    ERRIGNORE = 2>/dev/null
    DEVNUL := /dev/null
    WHICH := which
    SEP=/
endif

# Recursive wildcard 
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
MAKEFILE_LIST=Makefile
THIS_FILE := $(lastword $(MAKEFILE_LIST))
MERMAID_IMAGE=mikoto2000/mermaid.cli
# ENVIRONMENT Setting
DOCKER_ENV = true
CMD_ARGUMENTS ?= $(cmd)
ifeq ($(DOCKER_ENV),true)
    ifeq ($(shell ${WHICH} docker 2>${DEVNUL}),)
        $(error "docker is not in your system PATH. Please install docker to continue or set DOCKER_ENV = false in make file ")
    endif
    DOCKER_IMAGE ?= $(docker_image)
    DOCKER_CONTAINER_NAME ?=$(container_name)
    DOCKER_CONTAINER_MOUNT_POINT?=$(mount_point)
    ifneq ($(DOCKER_CONTAINER_NAME),)
        CONTAINER_RUNNING := $(shell docker inspect -f '{{.State.Running}}' ${DOCKER_CONTAINER_NAME})
    endif
    ifneq ($(DOCKER_CONTAINER_NAME),)
        DOCKER_IMAGE_EXISTS := $(shell docker images -q ${DOCKER_IMAGE} 2> /dev/null)
    endif
else
    ifeq ($(shell ${WHICH} yarn 2>${DEVNUL}),)
        $(error "yarn is not in your system PATH. Please install yarn to continue or set DOCKER_ENV = true in make file and use docker build pipeline ")
    endif
endif

# Mermaid Files
MERMAID_FILES?=$(patsubst %.mmd,%,$(subst mermaid/,, $(call rwildcard,mermaid/,*.mmd)))

.PHONY: all dep shell clean mermaid 
# ex : make cmd="ls -lah"
shell:
ifneq ($(DOCKER_ENV),)
ifeq ($(DOCKER_ENV),true)
    ifeq ($(DOCKER_IMAGE_EXISTS),)
	- @docker pull ${DOCKER_IMAGE}
    endif
    ifneq ($(CONTAINER_RUNNING),true)
	- @docker run --entrypoint "/bin/bash" -v ${CURDIR}:${DOCKER_CONTAINER_MOUNT_POINT} --name ${DOCKER_CONTAINER_NAME} --rm -d -i -t ${DOCKER_IMAGE} -c tail -f /dev/null
    endif
endif
endif
	- $(CLEAR)
ifneq ($(CMD_ARGUMENTS),)
    ifeq ($(DOCKER_ENV),true)
        ifneq ($(DOCKER_ENV),)
	- $(info Running in Docker Container)
	- @docker exec  --workdir ${DOCKER_CONTAINER_MOUNT_POINT} ${DOCKER_CONTAINER_NAME} /bin/bash -c "$(CMD_ARGUMENTS)"
        endif
    else
	- $(info Running in local environment)
	- @/bin/bash -c "$(CMD_ARGUMENTS)"
    endif
endif

mermaid:  clean
	- $(CLEAR)
	- $(MKDIR) fixtures/mermaid
    ifeq ($(DOCKER_ENV),true)
	for target in $(MERMAID_FILES); do \
			$(MAKE) --no-print-directory -f $(THIS_FILE) shell cmd="mmdc -p /opt/local/mermaid.cli/puppeteer-config.json -i mermaid/$$target.mmd -o fixtures/mermaid/$$target.png" docker_image="${MERMAID_IMAGE}" container_name="mermaid" mount_point="/home/node/data" ; \
	done
    else
    ifeq ($(shell ${WHICH} mmdc 2>${DEVNUL}),)
        $(error "mmdc (mermaid compiler) is not in your system PATH. Please run `make dep` to install dependancy ")
    endif
	for target in $(MERMAID_FILES); do \
			$(MAKE) --no-print-directory -f $(THIS_FILE) shell cmd="mmdc -i mermaid/$$target.mmd -o fixtures/mermaid/$$target.png"; \
	done
    endif
clean: 
	- $(CLEAR)
	- $(RM) fixtures/mermaid

dep: 
	- $(CLEAR)
	- sudo yarn global add mermaid.cli
