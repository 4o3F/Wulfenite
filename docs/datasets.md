# Dataset Notes

The TSE-specific training pipeline has been removed from `main`, but the
corpus assumptions below still hold for the next PSE phase.

## Retained Corpora

- `AISHELL-1`: primary clean Mandarin speech
- `AISHELL-3`: additional clean Mandarin speakers; resample to 16 kHz
- `MAGICDATA`: larger clean Mandarin speaker pool
- `CN-Celeb`: optional speaker-diversity expansion
- `MUSAN` or similar: additive non-speech noise

## Current Utility Coverage

The remaining Python utilities support:

- scanning the corpus layouts above into speaker-indexed dictionaries
- validating sample-rate assumptions for retained datasets
- resampling AISHELL-3 and CN-Celeb into the expected 16 kHz mono format
- scanning noise directories for augmentation pools

## Missing for the New PSE Phase

`main` does not yet include:

- a new PSE mixer
- emotion-aware reference or target mismatch generation
- evaluation splits for robustness to speaking-style drift

If emotion robustness becomes a formal target, add a Mandarin emotional
corpus such as ESD as a targeted supplement rather than replacing the
existing clean corpora.
