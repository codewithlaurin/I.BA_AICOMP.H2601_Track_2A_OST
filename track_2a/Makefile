.PHONY: build run test

# make run                      -> data/cases.jsonl, writes output/predictions.jsonl
# make run INPUT=path/cases.jsonl OUTPUT=path/predictions.jsonl
# Booklet paths in the cases file are relative to it; only the cases file and its
# booklets/ folder are mounted, so gold labels next to it never reach the container.
IMAGE ?= apertus-claimcheck:latest
PLATFORM ?= linux/amd64
INPUT ?= data/cases.jsonl
OUTPUT ?= output/predictions.jsonl
BOOKLETS ?= $(dir $(abspath $(INPUT)))booklets

# Endpoint: BASE_URL, API_KEY, LLM_NAME from the environment (a local .env is used if present).
ENV_VARS := BASE_URL API_KEY LLM_NAME LLM_BASE_URL LLM_API_KEY CSCS_INFERENCE_API_KEY APERTUS_MODEL

build:
	docker build --platform $(PLATFORM) -t $(IMAGE) .

run: build
	mkdir -p "$(dir $(abspath $(OUTPUT)))"
	docker run --rm --platform $(PLATFORM) \
		$(if $(wildcard .env),--env-file .env) \
		$(foreach var,$(ENV_VARS),-e $(var)) \
		-v "$(abspath $(INPUT)):/data/cases.jsonl:ro" \
		$(if $(wildcard $(BOOKLETS)),-v "$(BOOKLETS):/data/booklets:ro") \
		-v "$(dir $(abspath $(OUTPUT))):/output" \
		$(IMAGE) --input /data/cases.jsonl --output /output/$(notdir $(OUTPUT))

test:
	uv run python -m unittest discover -s tests
