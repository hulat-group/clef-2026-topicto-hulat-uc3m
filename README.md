# HULAT-UC3M at ImageCLEFtoPicto 2026

Reproducibility package for the HULAT-UC3M submission to the ImageCLEFtoPicto 2026 task: English text-to-pictogram generation with T5.

## Clone repository

Clone the repository with:

    git clone https://github.com/hulat-group/clef-2026-topicto-hulat-uc3m.git
    cd clef-2026-topicto-hulat-uc3m

The submitted checkpoint is stored with Git LFS. After cloning, run:

    git lfs install
    git lfs pull

Check that the model weights were downloaded correctly:

    ls -lh models/submitted_t5_base_checkpoint29640/model.safetensors

The file should be a large model file, approximately 892 MB, not a small Git LFS pointer.

## Environment

The experiments were run with:

- Python 3.10
- PyTorch 2.5.1+cu121
- Transformers 5.8.0
- Datasets 4.8.5
- NVIDIA GeForce RTX 3060, 12 GB

Create and activate a Python 3.10 environment:

    python -m venv .venv
    source .venv/bin/activate

Install the dependencies:

    python -m pip install --upgrade pip
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

## Data

The official ImageCLEFtoPicto task data are not redistributed in this repository.

To reproduce the submitted run, authorised users should place the official task files in:

    data/train.json
    data/valid.json
    data/test.json

The English ARASAAC lexicon used during output normalisation is included in this repository as:

    resources/arasaac_english.json

This file was obtained from the ARASAAC API endpoint:

    https://api.arasaac.org/v1/pictograms/all/en

More details about the expected data files are provided in:

    docs/DATA.md

## Generate predictions

After installing the environment, downloading the Git LFS checkpoint, and placing the official task data in `data/`, run:

    python src/generate_official_submission.py

The generated prediction file will be written to:

    outputs/official_submission.json

The script also writes intermediate raw and postprocessed predictions to `outputs/`.

## Output format

The generated file should contain one prediction for each test instance, using the official format:

    id
    hyp

## Training

The submitted model configuration can be trained with:

    python src/train_submitted_t5_base.py

The submitted system used T5-base, seed 777, learning rate 1e-4, maximum 20 epochs, early stopping patience 5, per-device batch size 2, gradient accumulation 8, effective batch size 16, maximum source length 64, maximum target length 32, and no mixed precision.

## Notes

The official submitted prediction file is not redistributed in this repository. The repository provides the code, environment specification, ARASAAC English lexicon, and Git LFS checkpoint needed to regenerate the prediction file locally using the official ImageCLEFtoPicto data.
