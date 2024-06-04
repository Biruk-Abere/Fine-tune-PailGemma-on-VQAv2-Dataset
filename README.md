# Fine-Tuning Pre-trained PaliGemma on VQAv2 Dataset

![Alt text](https://github.com/Biruk-Abere/Fine-tune-PailGemma-on-VQAv2-Dataset/blob/main/Screenshot%20from%202024-06-04%2017-53-47.png)


# Overview

This repository provides a step-by-step guide to fine-tuning Google's cutting-edge vision-language model, PaliGemma, on the Visual Question Answering v2 (VQAv2) dataset. PaliGemma is a powerful vision-language model from Google, which can take in both image and text inputs to generate text outputs. The model architecture consists of the SigLIPSo400m as the image encoder and Gemma-2B as the text decoder, making it highly versatile for various vision-language tasks

# Contents

 - Introduction
 - Installation
 - Dataset Preparation
 - Model Fine-Tuning
 - Inference
 - Parameter Efficient Fine-Tuning (PEFT)
 - Results
 - Acknowledgements

# Introduction

PaliGemma is a family of vision-language models with the architecture combining SigLIP for image encoding and Gemma for text decoding. The model supports three resolutions (224x224, 448x448, 896x896) and three precisions (bfloat16, float16, and float32). Fine-tuning these models can significantly improve their performance on specific tasks such as image captioning, visual question answering, object detection, and more.

# Installation

To run the fine-tuning process, ensure you have the following libraries installed:

    pip install transformers accelerate datasets
    pip install peft

# Dataset Preparation

We will use a subset of the VQAv2 dataset for fine-tuning. The dataset contains images with associated questions and answers.

    from datasets import load_dataset
    ds = load_dataset('HuggingFaceM4/VQAv2', split="train[:10%]")
    cols_remove = ["question_type", "answers", "answers_type", "image_id", "question_id"]
    ds = ds.remove_columns(cols_remove)
    split_ds = ds.train_test_split(test_size=0.05)
    train_ds = split_ds["test"]

# Model Fine-Tuning

The fine-tuning process involves several steps, including loading the pre-trained model, preparing the dataset, and setting up the training loop.

# Loading the Processor


    from transformers import PaliGemmaProcessor
    model_id = "google/paligemma-3b-pt-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id)

# Data Collation Function

    import torch

    device = "cuda"
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

    def collate_fn(examples):
        texts = ["answer " + example["question"] for example in examples]
        labels = [example['multiple_choice_answer'] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]
        tokens = processor(text=texts, images=images, suffix=labels,
                           return_tensors="pt", padding="longest",
                           tokenize_newline_separately=False)
        tokens = tokens.to(torch.bfloat16).to(device)
        return tokens

# Freezing Parts of the Model


    from transformers import PaliGemmaForConditionalGeneration
    import torch
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype = torch.bfloat16).to(device)
    
    for param in model.vision_tower.parameters():
      param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
      param.requires_grad = False

# Training Arguments 

      from transformers import TrainingArguments, Trainer
      
      args = TrainingArguments(
          num_train_epochs=2,
          remove_unused_columns=False,
          per_device_train_batch_size=4,
          gradient_accumulation_steps=4,
          warmup_steps=2,
          learning_rate=2e-5,
          weight_decay=1e-6,
          adam_beta2=0.999,
          logging_steps=100,
          optim="adamw_hf",
          save_strategy="steps",
          save_steps=1000,
          push_to_hub=True,
          save_total_limit=1,
          output_dir="paligemma_vqav2",
          bf16=True,
          report_to=["tensorboard"],
          dataloader_pin_memory=False
      )
      
      trainer = Trainer(
          model=model,
          train_dataset=train_ds,
          data_collator=collate_fn,
          args=args
      )
      
      trainer.train()
      trainer.push_to_hub()

  
# Parameter Efficient Fine-Tuning (PEFT)

PEFT techniques like LoRA allow fine-tuning large models on consumer hardware by adding and training only a small number of parameters while keeping the majority of the pre-trained model's parameters frozen.
# LoRA Configuration
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )
    
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

# Inference

To perform inference with the fine-tuned model:
      from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
      
      model_id = "merve/paligemma_vqav2"
      model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
      processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
      
      # Example inference
      from PIL import Image
      import requests
      
      prompt = "What is on the flower?"
      image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
      raw_image = Image.open(requests.get(image_url, stream=True).raw)
      inputs = processor(prompt, raw_image, return_tensors="pt")
      output = model.generate(**inputs, max_new_tokens=20)
      
      print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
