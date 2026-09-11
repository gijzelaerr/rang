# Message draft — not sent

Subject: Pointing solutions: a Gaussian-beam ambiguity worth checking?

Hi Oleg,

Following your suggestion to look at pointing solutions, I put together a
small Rust/JAX experiment with component-list DFT prediction and smooth
per-antenna pointing fits.

An interesting ambiguity appeared: for Gaussian beams, including rotating
elliptical ones, a particular smooth pointing change can be compensated by
sky-flux and antenna-gain changes, leaving every complex visibility unchanged.
I have an explicit finite transformation and numerical controls; one example
changes the pointing by about 35 arcseconds RMS with visibility differences
at floating-point roundoff.

It also explains why the initial restricted-trajectory fits looked much
better constrained. A central flux anchor alone doesn't remove these modes;
non-Gaussian beam structure and additional off-axis anchors change the picture.

I'm not claiming this is new—does the explicit pointing transformation ring
a bell? I'd value your take on whether a diagnostic separating data-driven
pointing information from prior constraints would be useful, and which
measured MeerKAT beam model would make the best next test.

Short note: https://github.com/gijzelaerr/rang/blob/main/research/review-brief.md

Derivation and reproducible controls:
https://github.com/gijzelaerr/rang/blob/main/research/gaussian-gauge.md

Gijs
