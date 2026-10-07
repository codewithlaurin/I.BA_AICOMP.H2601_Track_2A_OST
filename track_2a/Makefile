.PHONY: build run test

IMAGE ?= hackapertus-ost:dev
INPUT ?= data/cases.jsonl
OUTPUT_DIR ?= output
DATA_DIR ?= $(dir $(abspath $(INPUT)))
BOOKLETS_DIR ?= $(DATA_DIR)/booklets

build:
	docker build --platform linux/amd64 -t $(IMAGE) .

run: build
	mkdir -p "$(OUTPUT_DIR)"
	docker run --rm --platform linux/amd64 \
		-e BASE_URL -e API_KEY -e LLM_NAME \
		-e LLM_BASE_URL -e LLM_API_KEY \
		-e CSCS_INFERENCE_BASE_URL -e CSCS_INFERENCE_API_KEY -e APERTUS_MODEL \
		$(if $(wildcard $(BOOKLETS_DIR)),-v "$(abspath $(BOOKLETS_DIR)):/data/booklets:ro") \
		-v "$(abspath $(INPUT)):/data/cases.jsonl:ro" \
		-v "$(abspath $(OUTPUT_DIR)):/output" \
		$(IMAGE) --input /data/cases.jsonl --output /output/predictions.jsonl

test:
	uv run python -m unittest discover -s tests -v
