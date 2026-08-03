DOC := dayan_api_reference
MAIN := $(DOC).tex
CLASS := dayan-manual.cls
CHAPTERS := $(shell find chapters -type f -name '*.tex' -print | sort)
MANIFESTS := $(shell find symbols -type f -name '*.symbols' -print | sort)
COVERAGE_CHECK := scripts/check_public_api.sh
BUILD_DIR := build
REPO_ROOT := ..

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

check-api: $(COVERAGE_CHECK)
	@missing=0; \
	for manifest in $(MANIFESTS); do \
		while IFS= read -r symbol; do \
			test -z "$$symbol" && continue; \
			if ! grep -Fq "\code{$$symbol}" $(CHAPTERS); then \
				echo "error: public symbol $$symbol is not documented"; \
				missing=1; \
			fi; \
		done < $$manifest; \
	done; \
	while IFS= read -r symbol; do \
		if ! grep -Fq "\code{$$symbol}" chapters/05_dapi.tex; then \
			echo "error: exported dapi symbol $$symbol is not documented"; \
			missing=1; \
		fi; \
	done < $(REPO_ROOT)/device/dayan-api.symbols; \
	test $$missing -eq 0
	bash $(COVERAGE_CHECK)

check: $(PDF) check-api
	test -s $(PDF)
	pdfinfo $(PDF) | grep -q '^Pages:'
	pdffonts $(PDF) | awk 'NR > 2 { if ($$(NF-4) != "yes" || $$(NF-3) != "yes") exit 1 }'
	pdftotext $(PDF) $(TEXT)
	grep -Fq 'HAL API' $(TEXT)
	grep -Fq 'Service API' $(TEXT)
	grep -Fq 'Runtime API' $(TEXT)
	grep -Fq 'RPC API' $(TEXT)
	grep -Fq 'dapi API' $(TEXT)

clean:
	rm -rf $(BUILD_DIR)

FORCE:
