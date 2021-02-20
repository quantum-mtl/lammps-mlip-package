.PHONY: help \
	init-docker create-container start-container profile \
	test-all test-unit test-regression \
	clean-all clean-lammps clean-test \
	clean-docker clean-container clean-image
.DEFAULT_GOAL := help

###############################################################################
# Variables
###############################################################################
IMAGE_NAME = lammps
CONTAINER_NAME = lammps
DOCKERFILE = ./docker/Dockerfile
PWD = `pwd`

###############################################################################
# General Targets
###############################################################################
# ref: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

init-docker: ## initialie docker image
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

create-container: ## create docker container
	docker run -it -v $(PWD):/workspace --name $(CONTAINER_NAME) -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 $(IMAGE_NAME)

start-container: ## attach docker container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."
	docker attach $(CONTAINER_NAME)

stop-container: ## stop docker container
	docker stop $(CONTAINER_NAME)

profile: ## show profile of project
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"

test-all: test-unit test-regression  ## do unit and regression testing

test-unit:  ## do unit testing
	cd test; \
	mkdir -p build && cd build; \
	cmake ..; \
	make -j 32; \
	ctest -vv

test-regression: clean-lammps ## do regression test
	sh ./docker/install.sh
	@if [ -f ./lammps/src/lmp_serial ]; then \
		cd test/regression; \
			python3 regression.py; \
	else echo "message: lmp_serial does not exist!"; fi

clean-all: clean-lammps clean-test clean-docker  ## clean all artifacts

clean-lammps:  ## clean lammps-related binaries
	rm -f ./lammps/src/lmp_*
	rm -rf ./lammps/src/Obj_serial

clean-test:  ## clean test-related binaries
	rm -rf test/build test/Testing test/bin test/lib
	rm -f test/regression/dump.atom test/regression/log.lammps

clean-docker: clean-docker clean-image  # clean docker image and container

clean-container: ## remove docker container
	docker rm $(CONTAINER_NAME)

clean-image: ## remove docker image
	docker image rm $(IMAGE_NAME)
