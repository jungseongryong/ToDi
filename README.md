# ToDi: Token-wise Distillation via Fine-Grained Divergence Control (EMNLP 2025 Oral/🏆Outstanding Paper Award Finalist)

[![ACL Anthology](https://img.shields.io/badge/ACL-2025.emnlp--main.409-blue)](https://aclanthology.org/2025.emnlp-main.409/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-success)](https://aclanthology.org/2025.emnlp-main.409.pdf)
[![BibTeX](https://img.shields.io/badge/Paper-BibTeX-yellow)](#bibtex)

Official PyTorch implementation of **ToDi**, as presented in our paper:  
[**ToDi: Token-wise Distillation via Fine-Grained Divergence Control**](https://aclanthology.org/2025.emnlp-main.409/)  
**Seongryong Jung**, Suwan Yoon, DongGeon Kim, and Hwanhee Lee

---


Some of our code is based on [DSKD](https://github.com/songmzhang/DSKD), [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) and [Distillm](https://github.com/jongwooko/distillm/tree/master).

## 🔧 Requirements
- deepspeed >= 0.14.0
- torch >= 2.0.1
- transformers >= 4.40.2
- peft >= 0.8.2
- rouge_score >= 0.1.2


## 🤖 Models
You can download the corresponding model files (e.g., `pytorch_model.bin` or `model.safetensors`) of LLMs used in this paper into `model_hub/*/*/`.

Here are the links of these models on huggingface:
- GPT2-120M: [Here](https://huggingface.co/openai-community/gpt2)
- GPT2-1.5B (trained on Dolly by Gu et al.): [Here](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#31-resources)
- TinyLLaMA-1.1B: [Here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- Llama2-7B: [Here](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- OLMo2: [Here](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc)
- Qwen2.5: [Here](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- Gemma3: [Here](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)

## 🏋️ Training
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

## 📊 Evaluation
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



## 📚 BibTeX
If you find this repo useful for your research, please consider citing us:

```
@inproceedings{jung-etal-2025-todi,
    title = "{T}o{D}i: Token-wise Distillation via Fine-Grained Divergence Control",
    author = "Jung, Seongryong  and
      Yoon, Suwan  and
      Kim, DongGeon  and
      Lee, Hwanhee",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.409/",
    doi = "10.18653/v1/2025.emnlp-main.409",
    pages = "8089--8102",
    ISBN = "979-8-89176-332-6",
    abstract = "Large language models (LLMs) offer impressive performance but are impractical for resource-constrained deployment due to high latency and energy consumption. Knowledge distillation (KD) addresses this by transferring knowledge from a large teacher to a smaller student model. However, conventional KD, notably approaches like Forward KL (FKL) and Reverse KL (RKL), apply uniform divergence loss across the entire vocabulary, neglecting token-level prediction discrepancies. By investigating these representative divergences via gradient analysis, we reveal that FKL boosts underestimated tokens, while RKL suppresses overestimated ones, showing their complementary roles. Based on this observation, we propose Token-wise Distillation (ToDi), a novel method that adaptively combines FKL and RKL per token using a sigmoid-based weighting function derived from the teacher-student probability log-ratio. ToDi dynamically emphasizes the appropriate divergence for each token, enabling precise distribution alignment. We demonstrate that ToDi consistently outperforms recent distillation baselines using uniform or less granular strategies across instruction-following benchmarks. Extensive ablation studies and efficiency analysis further validate ToDi{'}s effectiveness and practicality."
}
```

## ✉️ Contact
If you have any questions or feedback, feel free to reach out:
- Seongryong Jung: jungsr1116@cau.ac.kr
