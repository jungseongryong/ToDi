# ToDi: Token-wise Distillation via Fine-Grained Divergence Control

Some of our code is based on [DSKD](https://github.com/songmzhang/DSKD), [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) and [Distillm](https://github.com/jongwooko/distillm/tree/master).

## Requirements
- deepspeed >= 0.14.0
- torch >= 2.0.1
- transformers >= 4.40.2
- peft >= 0.8.2
- rouge_score >= 0.1.2


## Models
You can download the corresponding model files (e.g., `pytorch_model.bin` or `model.safetensors`) of LLMs used in this paper into `model_hub/*/*/`.

Here are the links of these models on huggingface:
- GPT2-120M: [Here](https://huggingface.co/openai-community/gpt2)
- GPT2-1.5B (trained on Dolly by Gu et al.): [Here](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#31-resources)
- TinyLLaMA-1.1B: [Here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- Llama2-7B: [Here](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- OLMo2: [Here](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc)
- Qwen2.5: [Here](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- Gemma3: [Here](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)

## Training
### SFT for teacher models

For LLaMA2-7B (LoRA), run:
```bash
bash scripts/tinyllama/sft_teacher_llama2.sh
```


### SFT for student models
For GPT2-base (full fine-tuning), run:
```bash
bash scripts/gpt2/sft_gpt2_base.sh
```

For TinyLLaMA-1.1B (LoRA), run:
```bash
bash scripts/tinyllama/sft_tinyllama.sh
```


### KD 
For GPT2-base, run:
```bash
bash scripts/gpt2/vanilla_kd_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/vanilla_kd_tinyllama.sh
```

You can change the distance functions (e.g., KL Divergence, Reverse KL Divergence, JS Divergence, etc.) using `KD_OBJ` in the above scripts.






### File Structures in Output Directory
The output directory will be created under `./outputs` automatically after you run the training scripts. 
For full fine-tuning, the file structure of the output directory is as follows (take gpt2 SFT as an example):
```
./outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```
For LoRA fine-tuning, the file structure of the output directory is as follows (take TinyLLaMA LoRA SFT as an example):
```
./outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```

## Evaluation
### Evaluate Full Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval.sh ${CKPT_PATH} ${EVAL_BATCH_SIZE}
```
According to the above structure, `CKPT_PATH` is the **absolute path** of the model files like `/home/xxx/ToDi/outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../epochA_step...`.

### Evaluate LoRA Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval_lora.sh ${LORA_ADAPTER_PATH} ${EVAL_BATCH_SIZE}
```
Please note that `MODEL_PATH` in `run_eval_lora.sh` should be changed for different base models (TinyLLaMA, LLaMA2).

Similarly, `LORA_ADAPTER_PATH` is the **absolute path** of the LoRA adapter files like `/home/xxx/ToDi/outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../epochA_step...`.


