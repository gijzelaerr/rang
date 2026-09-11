# Rang working conventions

## Research scope

- Focus on MeerKAT. The current experiment is specified in `research/meerkat-scope.md`.
- Use simulations and comparisons to test research hypotheses; distinguish proposed methods, implemented methods and measured results.
- Preserve reproducible experiment settings, seeds, data provenance and relevant negative results.

## Implementation languages

- Use Rust for computational kernels and performance-sensitive simulation and calibration work.
- Use Python for astronomer-facing interfaces, experiment scripting and pipeline orchestration.
- Validate numerical correctness before claiming performance or scientific improvements.

## Git attribution

- Always use `gijsmolenaar@gmail.com` for non-Spotify or public work, including this repository. Do not use a Spotify email for Rang commits.
- Commits use the user's Git identity. Do not substitute an agent or bot identity.
- Never add agent signatures, AI attribution, generated-by footers or co-author trailers to commit messages.
- Follow the user's global Git and pre-commit verification rules.
