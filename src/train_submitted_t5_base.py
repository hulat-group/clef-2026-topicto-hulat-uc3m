from train_seq2seq_common import train_seq2seq_experiment


if __name__ == "__main__":
    train_seq2seq_experiment(
        step_number="submitted",
        step_name="t5_base_seed777_effbatch16_src_to_tgt",
        model_name="t5-base",
        model_output_name="submitted_t5_base_seed777_effbatch16",
        seed=777,
        learning_rate=1e-4,
        epochs=20,
        early_stopping_patience=5,
        batch_size=2,
        grad_accum=8,
        use_prefix=True,
        fp16=False,
    )
