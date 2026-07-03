import json
import math
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
)


class StopOnBadLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")

        bad_loss = loss is not None and (math.isnan(loss) or math.isinf(loss))
        bad_grad = grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm))

        if bad_loss or bad_grad:
            print("ERROR: Detectado loss/grad_norm inválido. Parando entrenamiento.")
            print(f"logs = {logs}")
            control.should_training_stop = True
            control.should_save = False


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {path}")


def normalize_src(text: str, use_prefix: bool):
    text = str(text).strip()
    if use_prefix:
        return "translate English to pictograms: " + text
    return text


def prepare_rows(rows, use_prefix: bool):
    data = []

    for row in rows:
        data.append({
            "id": row["id"],
            "src": row["src"],
            "tgt": row["tgt"],
            "input_text": normalize_src(row["src"], use_prefix=use_prefix),
            "target_text": row["tgt"],
        })

    return data


def inspect_labels(tokenized_dataset, n=10):
    lengths = []

    for i in range(min(n, len(tokenized_dataset))):
        labels = tokenized_dataset[i]["labels"]
        labels_no_pad = [x for x in labels if x != -100]
        lengths.append(len(labels_no_pad))

    return lengths


def train_seq2seq_experiment(
    step_number: str,
    step_name: str,
    model_name: str,
    model_output_name: str,
    seed: int,
    learning_rate: float,
    epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    grad_accum: int,
    use_prefix: bool,
    fp16: bool = False,
    max_source_length: int = 64,
    max_target_length: int = 32,
):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"
    models_dir = project_root / "models"

    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    train_path = data_dir / "train.json"
    valid_path = data_dir / "valid.json"
    model_output_dir = models_dir / model_output_name

    print("Loading data...")

    train_rows = load_json(train_path)
    valid_rows = load_json(valid_path)

    train_data = prepare_rows(train_rows, use_prefix=use_prefix)
    valid_data = prepare_rows(valid_rows, use_prefix=use_prefix)

    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)

    print(f"Train: {len(train_ds)}")
    print(f"Valid: {len(valid_ds)}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("🧹 Tokenizando...")

    tokenized_train = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    tokenized_valid = valid_ds.map(
        preprocess,
        batched=True,
        remove_columns=valid_ds.column_names,
    )

    label_lengths_preview = inspect_labels(tokenized_train, n=10)
    print(f"Preview longitud labels tokenizadas: {label_lengths_preview}")

    if any(x == 0 for x in label_lengths_preview):
        raise RuntimeError("Hay labels vacías tras tokenizar. Revisa target_text/tgt.")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_output_dir),

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,

        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,

        predict_with_generate=True,
        generation_max_length=max_target_length,

        fp16=fp16,
        bf16=False,
        gradient_checkpointing=True,

        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        seed=seed,

        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,

        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0001,
            ),
            StopOnBadLossCallback(),
        ],
    )

    print(" Entrenando experimento seq2seq...")

    train_result = trainer.train()

    print("📏 Evaluando mejor modelo cargado...")

    eval_metrics = trainer.evaluate()

    print(" Guardando modelo final...")

    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    trainer_state_path = model_output_dir / "trainer_state.json"
    trainer_state = load_json(trainer_state_path) if trainer_state_path.exists() else None

    best_checkpoint = None
    best_metric = None
    global_step = None
    epoch_final = None

    if trainer_state:
        best_checkpoint = trainer_state.get("best_model_checkpoint")
        best_metric = trainer_state.get("best_metric")
        global_step = trainer_state.get("global_step")
        epoch_final = trainer_state.get("epoch")

    payload = {
        "inputs": {
            "train": str(train_path),
            "valid": str(valid_path),
        },
        "model": {
            "base_model": model_name,
            "output_dir": str(model_output_dir),
            "task": "src_to_tgt_text",
            "use_prefix": use_prefix,
            "source_prefix": "translate English to pictograms: " if use_prefix else "",
            "best_checkpoint": best_checkpoint,
            "best_metric_eval_loss": best_metric,
        },
        "training_config": {
            "seed": seed,
            "learning_rate": learning_rate,
            "epochs_requested": epochs,
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "fp16": fp16,
            "label_lengths_preview": label_lengths_preview,
        },
        "run_result": {
            "global_step": global_step,
            "epoch_final": epoch_final,
            "stopped_before_requested_epochs": (
                epoch_final is not None and epoch_final < epochs
            ),
        },
        "train_metrics": train_result.metrics,
        "eval_metrics_best_model": eval_metrics,
        "trainer_state": trainer_state,
        "decision": {
            "next_step": "run_decoding_grid_for_this_model",
            "reason": "Modelo entrenado. Debe evaluarse con decoding grid y métricas oficiales/aproximadas.",
        },
    }

    out_path = output_dir / f"step_{step_number}_{step_name}.json"

    report = {
        "step": step_number,
        "name": step_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
    }

    save_json(out_path, report)

    print("\n Entrenamiento completado")
    print(json.dumps({
        "model_output_dir": str(model_output_dir),
        "best_checkpoint": best_checkpoint,
        "best_metric_eval_loss": best_metric,
        "global_step": global_step,
        "epoch_final": epoch_final,
        "eval_metrics_best_model": eval_metrics,
    }, ensure_ascii=False, indent=2))
