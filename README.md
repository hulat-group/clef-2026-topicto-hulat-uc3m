# HULAT-UC3M at ImageCLEFtoPicto 2026

Reproducibility package for the HULAT-UC3M submission to the ImageCLEFtoPicto 2026 task: English text-to-pictogram generation with T5.

## Environment

Tested with:

- Python 3.10.18
- PyTorch 2.5.1+cu121
- Transformers 5.8.0
- Datasets 4.8.5
- NVIDIA GeForce RTX 3060, 12 GB

## Installation

Clone the repository and make sure Git LFS downloads the model checkpoint:

    git lfs install
    git lfs pull

Create the environment:

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

## Repository contents

This repository includes:

- the training and inference scripts;
- the English ARASAAC lexicon used during output normalisation;
- the submitted T5 checkpoint, stored with Git LFS.

The checkpoint is expected at:

    models/submitted_t5_base_checkpoint29640/

The English ARASAAC lexicon is included at:

    resources/arasaac_english.json

It was obtained from the ARASAAC API endpoint:

    https://api.arasaac.org/v1/pictograms/all/en

## Data

The official ImageCLEFtoPicto task data are not redistributed in this repository.

To reproduce the submission, place the official task files in:

    data/train.json
    data/valid.json
    data/test.json

More details are provided in `docs/DATA.md`.

## Generate the submitted run

After installing the environment, downloading the Git LFS checkpoint, and placing the official task data in `data/`, run:

    python src/generate_official_submission.py

The generated prediction file will be written to:

    outputs/official_submission.json

The official submitted prediction file is not redistributed in this repository. The provided code, included ARASAAC lexicon, official task data, and submitted checkpoint allow the prediction file to be regenerated locally.

## Training

The submitted model configuration can be trained with:

    python src/train_submitted_t5_base.py
