# Data

The official ImageCLEFtoPicto 2026 task data are not redistributed in this repository.

To rerun training, validation decoding, or submission generation, authorised users should place the official task files in a local `data/` directory:

    data/train.json
    data/valid.json
    data/test.json

The expected fields are:

- `id`: instance identifier
- `src`: source sentence in English
- `tgt`: reference pictogram-term sequence, available for train and validation
- `pictos`: ARASAAC pictogram identifiers aligned with `tgt`, available for train and validation

The test file is expected to contain the instance identifier and the source sentence. The submitted system writes predictions in the official format with:

- `id`
- `hyp`

## ARASAAC lexicon

The English ARASAAC lexicon used by the output-normalisation step is included in:

    resources/arasaac_english.json

This file was obtained from the ARASAAC API endpoint:

    https://api.arasaac.org/v1/pictograms/all/en

It is not part of the official ImageCLEFtoPicto task data.

This normalisation step used only training/validation data and the English ARASAAC lexicon. It did not use test references.
