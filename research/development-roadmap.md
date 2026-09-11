# Quality and compute research gates

The objective is a demonstrable quality or efficiency advance over a relevant
state-of-the-art calibration/imaging method. Joint fitting, splines, automatic
differentiation, zero-mean constraints and QR elimination are not individually
novel. Our gain-only toy is a control, not a state-of-the-art comparison.

Work through these gates, preserving reproducible negative results:

1. Beam mismatch: controlled width, squint and antenna-dependent errors.
2. Gain flexibility: smooth frequency-dependent gains; quantify lost pointing information.
3. Out-of-family pointing: sinusoidal motion and jumps, beyond fitted splines.
4. Sky incompleteness: spectral mismatch, missing and extended sources.
5. Uncertainty coverage: repeated-noise tests and prior sensitivity.
6. Common pointing: map adequacy against true shared offset and external information.
7. Validation: contiguous gaps, baseline holdouts, initializations and separate tuning data.
8. Measured history: establish coordinate/record semantics before drift statistics.
9. Scaling: verified 64-antenna coordinates; measured runtime and peak memory.
10. Real observation: matched data/sky/beam comparison against an established pipeline.

For quality claims, compare visibility prediction, flux bias and image residuals
at matched compute. For efficiency claims, match output accuracy and report
hardware, warm/cold timings, convergence tolerance and peak memory. End-to-end
pipeline costs matter; a faster isolated kernel is insufficient.

Potential research direction, not an established result: use information after
accounting for uncertain sky and gains to decide which physical pointing/beam
parameters to fit, with predictive checks detecting inadequate models. This
must beat ordinary regularized joint fitting and relevant existing methods;
previous toy mode-selection tests did not demonstrate an accuracy advantage.

Historical data inspection (gate 8) already identified missing/repeated records;
external clarification is needed before a physically grounded drift model.
This does not block synthetic tests, but limits what we can claim about them.
