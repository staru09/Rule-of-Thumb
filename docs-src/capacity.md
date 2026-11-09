# Limits

Every explanation method trades something away, and an honest page
about those trades is worth more than a persuasive one. RoT's
simplicity — one shared, additive stand-in — is what makes it cheap,
query-free, and auditable from logged predictions alone. It is also,
inevitably, what blinds it to certain things. This page maps the
blind spots so you can work inside them with confidence.

## A model of averages, not arrangements

For text and images, a single shared set of weights is reused at
every position. That keeps the model tiny and position-independent,
but it means the stand-in effectively sees the *average word* of a
sentence and the *total ink per channel* of a picture:

```{image} _static/figures/limits-order.light.png
:class: only-light
:alt: Two sentences with swapped word order receiving identical explanations
```

```{image} _static/figures/limits-order.dark.png
:class: only-dark
:alt: Two sentences with swapped word order receiving identical explanations
```

- **Word order disappears.** "Dog bites man" and "man bites dog"
  receive the same explanation, because they contain the same words.
  Tasks driven by *which* words appear — sentiment, topic, resume
  screening — work beautifully; negation scope, syntax, and counting
  do not. If your question lives in the ordering, this is not the
  right tool, and it is better to learn that here than from a
  confusing result.

```{image} _static/figures/limits-ink.light.png
:class: only-light
:alt: A tight spot and a wide ring with equal total ink scoring the same
```

```{image} _static/figures/limits-ink.dark.png
:class: only-dark
:alt: A tight spot and a wide ring with equal total ink scoring the same
```

- **Position washes out on raw pixels.** A tight spot and a wide ring
  with equal total ink score the same, for exactly the same reason.
  This is why image paths travel through a backbone by default: rich
  feature maps (576 channels instead of 3) separate classes that bare
  pixels cannot. Think of raw pixels as a teaching aid for toy
  problems, not a production setting.

## Habits that keep you safe

1. **Feed it rich inputs**: transformer embeddings for text, backbone
   maps for images. The stand-in can only be as perceptive as what
   you show it.
2. **Always check agreement** (`predict` against the black box's own
   answers) before trusting a word of the explanation. Below about
   0.75, stop and reconsider — the fit will have warned you already.
3. **Narrow wide questions first.** Explaining a 1000-way classifier
   rarely means fitting all 1000 answers; remap to the question you
   actually care about (say, one class versus the rest) and fit on
   that:

```python
binary = (black_box_answers == 283).astype(np.int64)  # Persian-cat-vs-rest
exp = rot.fit_image(binary, images, mask=mask)
```

## Faithful is not the same as plausible

It helps to keep two judgements apart. RoT reproduces *the model's
answers* — that is fidelity, and the agreement score plus the reveal
curve measure it. Whether highlighted words look convincing to *a
person* is plausibility, a different question with its own baselines
(random orders, center-of-image priors, which embarrassingly often
win). A faithful ranking of what the model actually leans on can look
nothing like what you would underline yourself, especially where
averaging has discarded order or position. The
[Hatexplain](notebooks/04_hatexplain.ipynb) and
[Gaze](notebooks/05_salicon.ipynb) notebooks explore both sides
openly, baselines included.

## Why a whole reveal unit is dropped at a time

While it trains, RoT hides a random half of the input each step and
learns weights that still recover the model's answer from what is left
(the dropout rate is fixed at 0.5). The question is what "a piece of the
input" means: a whole reveal unit, or a single element inside one.

The package drops a whole reveal unit at a time, and never a single
element:

| Modality | What one draw hides | Shape of the keep mask | Legacy behaviour |
|---|---|---|---|
| Tabular | one feature | `(N, d)` | same (no divergence) |
| Text | one token, all its embedding dims | `(N, T)`, then the padding mask | matches Judicial `RoT_text`; MovieReview instead dropped per (token, element) |
| Image | one pixel, all its channels | `(N, H, W)`, then the padding mask | diverges: legacy dropped per (channel, pixel), `(N, C, H, W)` |

The reasons to keep whole-unit drops:

- Reveal curves hide a whole unit at a time when they score fidelity, so
  training should sample the same thing it will be judged on.
- Only whole-unit drops at 0.5 are uniform over the reveal subsets.
- Padding stays clean: the keep mask intersects the validity mask once.
- On images it is about 576 times fewer random draws.

The case for the old per-element drop was exact fidelity to two shipped
results (the pets-5000 and MovieReview experiments) and slightly finer
per-step noise. Measured, the difference is nothing: fitting the same
MobileNet feature maps both ways moved test fidelity by about 0.006,
within run-to-run noise, while switching would invalidate every committed
weight and test floor. So the package keeps the whole-unit drop.

## What the dropout rate and the starting `b` change in the explanation

Agreement alone can hide a change in the explanation, so a follow-up
compared the explanations themselves. It refits with only the dropout
rate or the starting value of `b` changed (the integration tests' fit
settings: 300 epochs, full batch, learning rate 0.05; three seeds per
setting) on two committed datasets: MobileNetV3-Small feature maps of the
1,797 digits (TinyCNN labels) and COMPAS (the SVC's labels). Everything
is measured on a held-out 20% against the default fit (rate 0.5, `b`
started at minus the mean). "Map corr" is the average per-example
correlation with the default fit's maps; "reveal gap" is the reveal-curve
area minus that of a random order; "seed corr" is the map correlation
between seeds.

| Data | Dropout rate | `b` start | Agreement | Map corr | Reveal gap | Seed corr |
|---|---|---|---|---|---|---|
| Digits (MobileNet) | 0 | −mean | 0.944 | 0.75 | −0.005 | 1.00 |
| Digits (MobileNet) | **0.5** | **−mean** | **0.954** | **0.90** | **+0.006** | **0.89** |
| Digits (MobileNet) | 0.75 | −mean | 0.936 | 0.87 | +0.004 | 0.86 |
| Digits (MobileNet) | 0 | 0 | 0.928 | 0.63 | −0.084 | 1.00 |
| Digits (MobileNet) | 0.5 | 0 | 0.961 | 0.71 | −0.045 | 0.99 |
| COMPAS | 0 | −mean | 0.950 | 0.91 | +0.143 | 1.00 |
| COMPAS | **0.5** | **−mean** | **0.892** | **0.99** | **+0.082** | **0.99** |
| COMPAS | 0.75 | −mean | 0.882 | 0.99 | +0.074 | 0.98 |

(The default row's map corr compares its other two seeds with the first.)

- **The dropout rate shapes the explanation more than the agreement.** On
  the digits, agreement stays within 0.94–0.96, but the maps fitted
  without dropout correlate only 0.75 with the default's, against 0.90
  between two default fits. Dropout also makes fits differ from seed to
  seed: a full-batch fit without it is deterministic.
- **On COMPAS, dropout costs fidelity.** More dropout lowers agreement
  (0.95 without, 0.89 at 0.5) and the reveal gap (+0.14 without, +0.08
  at 0.5), while the maps keep their pattern (correlation 0.91–0.99).
  0.5 is kept because it is uniform over reveal subsets, not because it
  scores best.
- **Starting `b` at minus the mean matters on images, and agreement does
  not show it.** Started at 0, the digit fits agree just as well
  (0.93–0.96), but their maps correlate only 0.63–0.71 with the default's
  and their reveal gap turns negative (down to −0.08): revealing in their
  order is worse than revealing at random. On standardised tabular data
  such as COMPAS the mean is already 0, so both starts give the same fit.
- **On 7×7 MobileNet maps the ranking is no better than random even for
  the default fit** (reveal gap within ±0.006): each cell's features see
  most of the digit, so any half of the cells carries about the same
  information.

An earlier check reran the research replication experiments (one seed)
with dropout off, or with `b` held at 0 for the whole fit (zeroed after
every step, so it never learns), using both the package defaults and the
old code's settings:

- **Tabular: dropout off copies the model better but changes the
  explanation.** Agreement rises (ScientificDiscovery 0.961 to 0.987,
  OpenXAI 0.973 to 0.987), but Blood Pressure enters the Pima top 3, the
  largest ScientificDiscovery Age importance grows from 1.12 to 8.65, and
  the OpenXAI baseline drops from 1.18 to 0.40 (paper 1.08).
- **`b` held at 0 breaks unscaled tabular data.** On ScientificDiscovery
  (raw values: age in years, glucose 0–199) agreement falls from 0.961 to
  0.766 (0.669 with the old settings). On standardised data it changes
  little, as on COMPAS above.
- **Text with the package defaults does better with `b` held at 0.**
  Text already starts `b` at 0; stopping it from learning raised the
  weighted AUROC on Judicial from 0.741 to 0.780 and on MovieReview from
  0.679 to 0.704. With the old code's much smaller learning rate `b`
  barely moves, so nothing changed.
- **Images agree with the table above.** Pets moved by at most 0.002
  with dropout off, within seed noise. On the digits, `b` held at 0
  raised the reveal-curve area (0.839 to 0.855), but only because the
  random order's area rose more (0.859 to 0.895): the reveal gap fell
  from −0.020 to −0.040, the same loss as starting `b` at 0.
