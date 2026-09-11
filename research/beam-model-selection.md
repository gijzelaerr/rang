# Beam-model selection needs a science-quality control

We compare fixed beam, one fitted log-width, log-width plus frequency slope,
and all three of log-width, slope and log-axis ratio. This is an ordinary
joint-fit/model-selection baseline, **not a new algorithm**.

The beam callback uses log geometric FWHM correction a+b*x and log(y/x)
correction q, where x=(frequency-1.3GHz)/0.4GHz. Individual axes receive
corrections a+b*x-q/2 and a+b*x+q/2. Squint is unchanged. Solver argument
`beam_shape_prior=[0.03,0.03,0.03]` gives independent zero-centred priors;
zero entries fix parameters. It cannot be combined with the scalar width prior.

## Independent evaluation

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/beam_model_selection.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/beam_model_selection.py --rich
```

Twelve seeds each use a new pointing/gain truth and noise realization, with
the same realization paired across truth-beam ablations. Complete times are
split into training, validation (indices1,7,13,19) and test (4,10,16,22).
Each candidate uses training data only. Lowest validation residual among
converged fits chooses a candidate; test data are not used in that decision.
There is no refit after selection. Priors, knots, splits and candidate families
are fixed before evaluating the test set. All288 fits across both campaigns
converge. [Individual results](results/beam-model-selection.json).

## A lower residual can conceal worse fluxes

For true chromatic width error, the single-width fit has mean test residual
statistic1.288 and mean maximum flux-ratio bias **1.10%**. The fixed model has
larger residuals but much smaller flux-ratio bias in this case. Validation
therefore rewards a visibility improvement that is not a science-quality win.
Here flux ratios are relative to the central source, cancelling the global
gain/flux-scale ambiguity. No external flux anchor is imposed in these fits.

The richer candidate set resolves these controlled mismatches:

| True perturbation | Mean selected test residual statistic | Mean maximum flux-ratio error |
| --- | --- | --- |
| Matched beam | 1.041 | 0.0144% |
| Uniform width +1% | 1.042 | 0.0173% |
| Axis log-width corrections (-0.01,+0.01) | 1.042 | 0.0318% |
| Width multiplied by 1+0.01*x | 1.042 | 0.0334% |

All axis-ratio cases select the three-parameter model. Chromatic cases split
equally between the two- and three-parameter models. Matched cases select
fixed7/12, width2/12, three-parameter3/12. Uniform-width cases select
width7/12, two-parameter1/12, three-parameter4/12.

Always fitting the richest model also handles these errors, but has somewhat
higher flux-ratio variance in the matched/uniform-width cases. No robust
superiority of selection over that baseline has been established. Enumerating
all candidates costs their combined fits, so it is not an efficiency result.
Recorded timings include compilation and are not a controlled benchmark.

These are low-dimensional, largely in-family synthetic beam errors. The
chromatic truth is linear in width rather than log-width, leaving a small
second-order mismatch. Antenna-dependent beams, squint, higher-order shape,
missing emission and real observations remain essential tests. Future
adaptive proposals must beat these ordinary joint-fit baselines on both
science quality and resource use, not only compare to an incorrect fixed beam.
