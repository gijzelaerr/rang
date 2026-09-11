# Message draft — not sent

Subject: MeerKAT pointing experiments: beam errors and next comparison

Hi Oleg,

Following your pointing suggestion and zero-mean feedback, Rang now has a
Rust/JAX experiment with component-list DFT prediction and smooth per-antenna
relative pointing, separate from an optional prior-constrained common offset.

The strongest new control is dish-to-dish beam width: synthetic 1% log-width
scatter gives roughly 6–9 arcsec relative pointing errors even when a shared
width is fitted. Adding seven relative widths brings this to 0.8–1.1 arcsec,
close to the exact-known-beam control. These are eight-antenna, matched-family
simulations, not measured MeerKAT beam errors or a claim of novel joint fitting.

Relative-pointing interval coverage improves substantially in a small noise
pilot, but common-mode coverage remains weaker. The exact Gaussian ambiguity
and the earlier negative mode-selection result are retained in the repository.

The next hypothesis is to select pointing/beam freedom using separate limits
on faint-signal distortion and model-error contamination, then test whether
this beats a fairly tuned joint fit in quality or total computation. This is
still a proposal. Which existing solver, measured beam errors and science-field
dataset would make the most useful comparison? Is this selection question worth
pursuing beyond established joint calibration and solution-interval methods?

Overview and current limitations:
https://github.com/gijzelaerr/rang

Controlled beam-width result:
https://github.com/gijzelaerr/rang/blob/main/research/antenna-beam-widths.md

Gijs
