FROM vllm/vllm-openai:v0.33.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /code
RUN mkdir -p /code/eval

# Data-processing dependencies; vLLM and torch come from the base image.
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    openpyxl==3.1.5 \
    python-dateutil==2.9.0.post0

COPY src/ /code/src/
COPY eval/run_eval.py /code/eval/run_eval.py
COPY eval/run_eval_llm.py /code/eval/run_eval_llm.py
COPY eval/holdout.py /code/eval/holdout.py
COPY predict.py /code/predict.py
COPY docker/README.code.md /code/README.md
COPY data_manifest.json /code/data_manifest.json
COPY docker/predict.sh /code/predict.sh
COPY docker/training_code/ /code/training_code/
COPY data/raw/train/ /code/data/raw/train/
COPY qa/ /code/qa/

# Model weights are baked into the image: inference must run fully offline.
COPY models/gemma-4-12b-it/ /code/models/gemma-4-12b-it/

RUN chmod +x /code/predict.sh

# The base image starts the vLLM server; reset it so plain docker run starts us.
ENTRYPOINT []
CMD ["/code/predict.sh"]
