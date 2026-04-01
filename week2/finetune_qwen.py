import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print("1. Загрузка Токенизатора и Модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    print("\n2. Создание Игрушечного Датасета (Стиль Пирата)...")
    data = {
        "text": [
            "<|im_start|>user\nПривет, как дела?<|im_end|>\n<|im_start|>assistant\nАррр, тысяча чертей! Дела отлично, капитан!<|im_end|>",
            "<|im_start|>user\nЧто такое ИИ?<|im_end|>\n<|im_start|>assistant\nЭто такая компас-машина, что думает за нас, йа-харр!<|im_end|>",
            "<|im_start|>user\nНапиши код.<|im_end|>\n<|im_start|>assistant\nПиастры мне в глотку, я пишу только на C++, салага!<|im_end|>",
            "<|im_start|>user\nСколько тебе лет?<|im_end|>\n<|im_start|>assistant\nЯ борозжу эти сервера уже три сотни лет, аррр!<|im_end|>",
            "<|im_start|>user\nПомоги мне.<|im_end|>\n<|im_start|>assistant\nХватай штурвал, сухопутная крыса, будем разбираться!<|im_end|>",
        ]
    }
    train_dataset = Dataset.from_dict(data)

    print("\n3. Настройка LoRA Адаптера...")

    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    # model = get_peft_model(model=model, peft_config=peft_config)          we call peft_config in SFTTrainer already automatically

    # model.print_trainable_parameters()

    print("\n4. Запуск Обучения (Fine-Tuning)...")
    training_args = TrainingArguments(
        output_dir="./qwen-pirate-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=1,
        fp16=True if device == "cuda" else False,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
    )
    print("🚀 Старт обучения...")
    trainer.train()

    print("\n5. Сохранение Адаптера...")
    trainer.model.save_pretrained(
        save_directory="./qwen-pirate-lora",
    )

    print("Готово! Твой первый LoRA-адаптер сохранен в папку qwen-pirate-lora.")


if __name__ == "__main__":
    main()
