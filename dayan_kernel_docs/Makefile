DOC := dayan_kernel_manual
MAIN := $(DOC).tex
CLASS := dayan-manual.cls
CHAPTERS := $(shell find chapters -type f -name '*.tex' -print | sort)
BUILD_DIR := build
REPO_ROOT := ..
PUBLIC_SYMBOLS := $(REPO_ROOT)/dayan-api.symbols
API_CHAPTERS := chapters/04_dayan_api.tex chapters/05_libc.tex

GIT_TAG := $(shell git -C $(REPO_ROOT) describe --tags --exact-match HEAD 2>/dev/null)
GIT_COMMIT := $(shell git -C $(REPO_ROOT) rev-parse --short=8 HEAD 2>/dev/null || printf unknown)
GIT_DIRTY := $(shell if git -C $(REPO_ROOT) rev-parse --is-inside-work-tree >/dev/null 2>&1 && test -n "$$(git -C $(REPO_ROOT) status --porcelain --untracked-files=normal --ignored=no)"; then printf -- "-dirty"; fi)
GIT_VERSION := $(if $(GIT_TAG),$(GIT_TAG),$(GIT_COMMIT))$(GIT_DIRTY)
FILE_VERSION := $(subst /,_,$(GIT_VERSION))

VERSION_TEX := $(BUILD_DIR)/version.tex
VERSIONED_PDF := $(BUILD_DIR)/$(DOC)-$(FILE_VERSION).pdf
PDF := $(BUILD_DIR)/$(DOC).pdf
TEXT := $(BUILD_DIR)/$(DOC).txt

.PHONY: all pdf check check-api clean FORCE

all: pdf

pdf: $(PDF)

$(BUILD_DIR):
	mkdir -p $@

$(VERSION_TEX): FORCE | $(BUILD_DIR)
	@tmp=$@.tmp; \
	printf '%s\n' '\providecommand{\DocVersion}{\detokenize{$(GIT_VERSION)}}' > $$tmp; \
	if test -r $@ && cmp -s $$tmp $@; then rm -f $$tmp; else mv $$tmp $@; fi

$(VERSIONED_PDF): $(MAIN) $(CLASS) $(CHAPTERS) $(VERSION_TEX) | $(BUILD_DIR)
	latexmk -xelatex \
		-interaction=nonstopmode \
		-halt-on-error \
		-file-line-error \
		-outdir=$(BUILD_DIR) \
		-jobname=$(notdir $(basename $@)) \
		$(MAIN)

$(PDF): $(VERSIONED_PDF)
	cp $< $@

check-api:
	@missing=0; \
	while IFS= read -r symbol; do \
		if ! grep -Fq "\code{$$symbol}" $(API_CHAPTERS); then \
			echo "error: public symbol $$symbol is not documented"; \
			missing=1; \
		fi; \
	done < $(PUBLIC_SYMBOLS); \
	test $$missing -eq 0

check: $(PDF) check-api
	test -s $(PDF)
	pdfinfo $(PDF) | grep -q '^Pages:'
	pdffonts $(PDF) | awk 'NR > 2 { if ($$(NF-4) != "yes" || $$(NF-3) != "yes") exit 1 }'
	pdftotext $(PDF) $(TEXT)
	grep -Fq 'Dayan API' $(TEXT)
	grep -Fq '0x00400000' $(TEXT)
	grep -Fq 'rv64imaf_zicsr_zifencei' $(TEXT)
	grep -Fq 'npu-os-api.elf' $(TEXT)
	grep -Fq 'dycc' $(TEXT)
	grep -Fq 'npu_kernel_launch_attrs_t' $(TEXT)
	grep -Fq '软件栈架构' $(TEXT)
	grep -Fq 'Dayan OS 内部机制' $(TEXT)
	grep -Fq 'Host RPC 服务' $(TEXT)
	grep -Fq 'RPC 传输与处理' $(TEXT)

clean:
	rm -rf $(BUILD_DIR)

FORCE:
