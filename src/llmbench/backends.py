class HFBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

        self.torch = torch
        self.device = args.device
        if args.quantization == "int8" and args.device != "cpu":
            raise ValueError("Dynamic INT8 is a CPU experiment; use --device cpu")
        if args.quantization == "nf4" and args.device != "cuda":
            raise ValueError("This NF4 profile requires an NVIDIA CUDA GPU")
        if args.quantization == "int8" and args.dtype != "float32":
            raise ValueError("Dynamic INT8 starts from --dtype float32")
        kwargs = dict(revision=args.revision, torch_dtype=getattr(torch, args.dtype))
        if args.quantization == "nf4":
            kwargs.update(
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, args.dtype),
                    bnb_4bit_use_double_quant=True,
                ),
            )
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
        if args.quantization == "int8":
            import platform

            if (
                platform.machine().lower() in ("arm64", "aarch64")
                and "qnnpack" in torch.backends.quantized.supported_engines
            ):
                torch.backends.quantized.engine = "qnnpack"
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
        if args.quantization != "nf4":
            self.model.to(args.device)
        self.config = GenerationConfig(
            max_new_tokens=args.output_tokens,
            min_new_tokens=args.output_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
        )
        self.output_tokens = args.output_tokens

    def prepare(self, prompts):
        # All prompts have equal length, so no padding or masks of padded tokens.
        ids = self.torch.tensor(prompts, dtype=self.torch.long, device=self.device)
        return {"input_ids": ids, "attention_mask": self.torch.ones_like(ids)}

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def generate(self, inputs):
        with self.torch.inference_mode():
            return self.model.generate(**inputs, generation_config=self.config)

    def extract(self, raw, inputs):
        return raw[:, inputs["input_ids"].shape[1] :].cpu().tolist()


class VLLMBackend:
    def __init__(self, args):
        if args.device != "cuda" or args.quantization != "none":
            raise ValueError("vLLM profile requires --device cuda --quantization none")
        from vllm import LLM, SamplingParams

        self.model = LLM(
            model=args.model,
            revision=args.revision,
            tokenizer_revision=args.revision,
            dtype=args.dtype,
            max_model_len=args.input_tokens + args.output_tokens + 16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
            seed=args.seed,
            generation_config="vllm",
        )
        self.params = SamplingParams(
            temperature=0,
            max_tokens=args.output_tokens,
            min_tokens=args.output_tokens,
            ignore_eos=True,
            detokenize=False,
        )

    def prepare(self, prompts):
        return [{"prompt_token_ids": ids} for ids in prompts]

    def sync(self):
        # LLM.generate blocks until every request completes, including GPU work.
        pass

    def generate(self, inputs):
        return self.model.generate(inputs, self.params, use_tqdm=False)

    def extract(self, raw, inputs):
        return [list(item.outputs[0].token_ids) for item in raw]
