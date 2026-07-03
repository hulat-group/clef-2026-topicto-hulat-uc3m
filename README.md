# HULAT-UC3M at ImageCLEFtoPicto 2026

Code and selected artifacts for the HULAT-UC3M submission to the ImageCLEFtoPicto 2026 task.

## Environment

Tested with Python 3.10.18, PyTorch 2.5.1+cu121, Transformers 5.8.0, Datasets 4.8.5, and an NVIDIA GeForce RTX 3060 GPU with 12 GB of memory.

## Installation

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

## Reproducibility scope

The English ARASAAC lexicon used during output normalisation is included as `resources/arasaac_english.json`. This file was obtained from the ARASAAC API endpoint `https://api.arasaac.org/v1/pictograms/all/en`.

This repository provides the scripts, validation summary, and official submitted files used for the HULAT-UC3M run.

The official task data are not redistributed. To rerun training or inference, place the task files in:

    data/train.json
    data/valid.json
    data/test.json
    data/arasaac_english.json

The selected checkpoint should be placed in:

    models/submitted_t5_base_checkpoint29640/

## Training

    python src/train_submitted_t5_base.py

## Generate official submission

    python src/generate_official_submission.py

The final submission was generated with greedy decoding. Before writing the generated submission file, a minor output-normalisation step was applied using only training/validation data and the English ARASAAC lexicon. It did not use test references and affected 23 out of 4,306 test predictions.


More details about the expected data files are provided in `docs/DATA.md`.


## Reproduce the official submission from the released checkpoint

After installing the environment, place the official ImageCLEFtoPicto files in:

    data/train.json
    data/valid.json
    data/test.json

The English ARASAAC lexicon used during output normalisation is already included as:

    resources/arasaac_english.json

This file was obtained from the ARASAAC API endpoint:

    https://api.arasaac.org/v1/pictograms/all/en

Download the released checkpoint and place it in:

    models/submitted_t5_base_checkpoint29640/

Then regenerate the official submission with:

    python src/generate_official_submission.py

The generated file will be written to:

    outputs/official_submission.json

The official submitted prediction file is not redistributed in this repository to avoid releasing task-derived prediction outputs. The released checkpoint and script allow the submitted run to be regenerated locally.
