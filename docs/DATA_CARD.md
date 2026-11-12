# Data card

## TDL

SISO OFDM uses 264 subcarriers, 14 symbols, 15 kHz spacing, a 16-sample cyclic
prefix, pilots at symbols 2 and 11 (zero-based), 4 GHz carrier, 16-QAM, and 5G LDPC
with 8,142 information bits and 12,672 coded bits per frame. The decoder uses the
original Sionna configuration. `link.py` retains the original implementation's
semantic fingerprint.

Source A uses 50 ns delay spread and speeds 0–1.5 m/s. Source B uses 150 ns and
0–3 m/s. The standard target B uses 500 ns and 0–5 m/s; target C uses 1,000 ns and
0–10 m/s. `configs/datasets.json` specifies the delay and speed variants.

Source training has 200 stored batches per SNR from 4 to 18 dB. Standard online
training has ten batches per SNR from 4 to 20 dB. Each stored batch contains 32
frames. Validation and PEBS evaluation each use 100 batches at 12 dB. Main SNR
curves keep bits/channel fixed and scale the recorded AWGN variance. Main TDL
curves use the validation samples; PEBS uses a separate evaluation stream.
Independent final evaluation uses seed20261110, with no reselection or training.

The original A2B online/validation/PEBS files are distributed as a versioned
Release asset, verified against their original SHA256 values. Regeneration uses
explicit seeds and recipes. Exact sampled arrays also depend on generator call
order and software version. Results from newly generated inputs are saved as a
separate run, never substituted into the archived paper CSVs.

## RT

`artifacts/rt/` contains 512 source propagation responses (128 per city) and nine
unique fixed target responses. Some target responses are reused in cross-city
comparisons. The target geometry is fixed; different payload, noise and
interference realizations are generated using the original per-frame seed rule.
This is not 3,200 independently selected propagation positions.

Source pretraining uses 50 signal realizations per link and SNR, giving 96,000
frames per epoch. Target adaptation cycles through 4–18 dB, restarting the SNR
sequence at the demapper stage. Validation uses 256 frames at 12 dB; final testing
uses 3,200 frames per SNR. Target narrowband QPSK interference occupies 33 of 264
subcarriers starting at index116, with SIR15 dB. Source data have no added
interference. The delay cases rescale the target path delays with the remaining
propagation and interference settings fixed. The released arrays freeze the
resulting channel responses.

RT frame generation and model execution use independent validation/test roles.
Exact internal position IDs are preserved because they enter the random seed.
The documentation maps those IDs to the paper's P1/P2 labels.

## Integrity

Input manifests record role, SNR, relative file path and SHA256. Path traversal,
unexpected pickle globals, cross-role sample overlap, invalid channels and
nonfinite input arrays are rejected. Only known NumPy-array pickle constructors
are accepted. All downloaded archives are verified before extraction; symlinks
and files outside the output root are rejected.
