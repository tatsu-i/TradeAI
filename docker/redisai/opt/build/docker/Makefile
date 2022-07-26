
.NOTPARALLEL:

ROOT=../../..

ifeq ($(VERSION),)
VERSION:=$(patsubst v%,%,$(shell git describe --tags `git rev-list --tags --max-count=1`))
endif
ifeq ($(VERSION),)
$(error Cannot determine version. Aborting.)
endif

# OSNICK=buster|stretch|bionic
OSNICK ?= buster

REPO=redisfab
STEM=$(REPO)/redisai-cpu
BUILD_OPT=--rm
# --squash

FETCH_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm64v8=WITH_ORT=0 WITH_PT=0

FETCH_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0
BUILD_ARGS.arm32v7=WITH_ORT=0 WITH_PT=0

#----------------------------------------------------------------------------------------------

define targets # (1=OP, 2=op)
$(1)_TARGETS :=
$(1)_TARGETS += $(if $(findstring $(X64),1),$(2)_x64)
$(1)_TARGETS += $(if $(findstring $(ARM7),1),$(2)_arm32v7)
$(1)_TARGETS += $(if $(findstring $(ARM8),1),$(2)_arm64v8)

$(1)_TARGETS += $$(if $$(strip $$($(1)_TARGETS)),,$(2)_x64 $(2)_arm32v7 $(2)_arm64v8)
endef

$(eval $(call targets,BUILD,build))
$(eval $(call targets,PUSH,push))

#----------------------------------------------------------------------------------------------

define build_x64 # (1=arch)
build_$(1):
	docker build $(BUILD_OPT) -t $(STEM)-$(OSNICK):$(VERSION)-x64 -f $(ROOT)/Dockerfile \
		--build-arg DEPS_ARGS="$(FETCH_ARGS.x64)" \
		--build-arg BUILD_ARGS="$(BUILD_ARGS.x64)" \
		$(ROOT)

.PHONY: build_$(1)
endef

define build_arm  # (1=arch)
build_$(1):
	docker build $(BUILD_OPT) -t $(STEM)-$(OSNICK):$(VERSION)-$(1) -f $(ROOT)/Dockerfile.arm \
		--build-arg ARCH=$(1) \
		--build-arg DEPS_ARGS="$(FETCH_ARGS.$(1))" \
		--build-arg BUILD_ARGS="$(BUILD_ARGS.$(1))" \
		$(ROOT)

.PHONY: build_$(1)
endef

define push # (1=arch)
push_$(1):
	docker push $(STEM)-$(OSNICK):$(VERSION)-$(1)

.PHONY: push_$(1)
endef

define create_manifest # (1=version)
docker manifest create -a $(STEM)-$(OSNICK):$(1) \
	-a $(STEM)-$(OSNICK):$(VERSION)-x64 \
	-a $(STEM)-$(OSNICK):$(VERSION)-arm64v8 \
	-a $(STEM)-$(OSNICK):$(VERSION)-arm32v7
docker manifest annotate $(STEM)-$(OSNICK):$(1) $(STEM)-$(OSNICK):$(VERSION)-arm32v7 --os linux --arch arm --variant v7
docker manifest annotate $(STEM)-$(OSNICK):$(1) $(STEM)-$(OSNICK):$(VERSION)-arm64v8 --os linux --arch arm64 --variant v8
docker manifest push -p $(STEM)-$(OSNICK):$(1)
endef

#----------------------------------------------------------------------------------------------

all: build publish

$(eval $(call build_x64,x64))
$(eval $(call build_arm,arm64v8))
$(eval $(call build_arm,arm32v7))

$(eval $(call push,x64))
$(eval $(call push,arm64v8))
$(eval $(call push,arm32v7))

build: $(BUILD_TARGETS)

ifneq ($(PUSH),0)
publish: $(PUSH_TARGETS)
else
publish:
endif
	$(call create_manifest,$(VERSION))
	$(call create_manifest,latest)
	
show:
	@echo "$(STEM)-$(OSNICK):"
ifeq ($(INSPECT),1)
	@docker manifest inspect $(STEM)-$(OSNICK):$(VERSION) | jq
else
	@curl -s -X "GET" https://cloud.docker.com/v2/repositories/$(STEM)-$(OSNICK)/tags/ | jq
endif

.PHONY: build public show
