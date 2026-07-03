# Data

The official ImageCLEFtoPicto 2026 task data are not redistributed in this repository.

To rerun training or generate predictions, authorised users should place the official task files in a local `data/` directory:

    data/train.json
    data/valid.json
    data/test.json

The expected fields are:

- `id`: instance identifier
- `src`: source sentence in English
- `tgt`: reference pictogram-term sequence, available for training and validation
- `pictos`: ARASAAC pictogram identifiers aligned with `tgt`, available for training and validation

The test file is expected to contain the instance identifier and the source sentence. The generated prediction file uses the official submission format:

- `id`
- `hyp`

## ARASAAC English lexicon

The English ARASAAC lexicon used by the output-normalisation step is included in:

    resources/arasaac_english.json

This file was obtained from the ARASAAC API endpoint:

    https://api.arasaac.org/v1/pictograms/all/en

It is not part of the official ImageCLEFtoPicto task data. It was used only by the output-normalisation step. This step uses only training/validation data and the English ARASAAC lexicon; it does not use test references.
