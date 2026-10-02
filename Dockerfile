# Target platform: Linux x86_64 with an NVIDIA GPU. Not a macOS GPU image.
FROM vllm/vllm-openai:v0.10.2
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python3 -m pip install --no-cache-dir . "bitsandbytes==0.45.5" "nvidia-ml-py==12.570.86"
ENTRYPOINT ["python3", "-m", "llmbench"]
CMD ["--help"]
