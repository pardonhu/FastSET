# Changelog

## 1.0.0 — 2026-11-12

- Release under the MIT license with updated manuscript metadata and authors.
- Provide 152 explicit experiment configurations and a shared TDL/RT execution
  package for generation, source pretraining, adaptation, evaluation and timing.
- Include final paper, supplement and difference-summary plotting sources,
  48 numerical CSV tables and a figure-to-experiment index.
- Publish 101 model states and the original A2B input files as verified Release
  assets; include 521 frozen RT channel-response arrays in the repository.
- Record configuration, source/data/code hashes, batch sequences and integer error
  counts in isolated result directories.
- Add strict initial/iteration BER stopping, full-trajectory PEBS search,
  deterministic stage ordering, frame-based batch sizing and resident timing.
- Verify a fresh CPU installation with 30 tests and 53 generated PDF outputs.
  Validate the GPU branches and full A2B run at the scales listed in
  [the validation record](docs/RELEASE_STATUS.md).
