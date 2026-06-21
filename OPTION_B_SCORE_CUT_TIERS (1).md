# Option B: cut the tiers on the displayed score

*12 Jun 2026. Written after Guney spotted, via the new tier-range legend,
that the supervised tiers overlap on the score shown in the dashboard.
Every number below is recomputed from the committed final parquets
(`team_model/london_final_tiers.parquet`,
`england_model/england_final_tiers.parquet`). This is a proposal for the
group — the tier labels are Natalia's Phase 6 output, so nothing here gets
implemented without her sign-off.*

## The problem

The dashboard now prints each tier's risk-score range in the legend, and
the ranges overlap. London, static index:

| tier | label | score range | LSOAs |
|---|---|---|---|
| 1 | Highest | 75.1 – 100.0 | 1,265 |
| 2 | High | 73.2 – 77.1 | 616 |
| 3 | Moderate | 68.9 – 75.2 | 1,247 |
| 4 | Low | 0.0 – 71.3 | 1,866 |

Concrete pair: Havering 004A scores 75.49 and is Tier 1; Enfield 023B
scores 75.50 higher and is Tier 2. Roughly 600 LSOAs sit inside each
boundary's overlap band. England is worse on paper (Tier 2 spans 24.3–57.2
against Tier 1's floor of 41.0) because the structural score is flatter.

This is not a bug. The Phase 6 thresholds were placed on the **24-month
training score** (ROC/Youden against the LSOAs that became top-15% demand
in the following 12 months), while the dashboard displays the **36-month
score**. Any LSOA whose last 12 months ran hotter or colder than its first
24 drifts away from the score that decided its tier, and near a cut-point
that drift interleaves the labels.

The problem is operational. The dashboard's job is to hand a
recommendation to a police audience, and "why is this one Highest when
that one has the same score?" is a question the tool invites and cannot
answer inside the briefing. A legend that needs a methodology footnote to
not look self-contradictory is losing the room.

## What Option B changes

Apply the calibrated thresholds to the score the user is looking at. Keep
tier sizes exactly as Phase 6 produced them (rank cut: the top 1,265
London LSOAs by displayed score are Tier 1, the next 616 Tier 2, and so
on), which on the current data is equivalent to fixed score boundaries:

- **London**: Highest ≥ 76.2, High ≥ 74.0, Moderate ≥ 69.8, Low below.
  298 of 4,994 LSOAs change label (6.0%), every one by exactly one tier,
  in symmetric swaps at each boundary.
- **England**: Highest ≥ 44.6, High ≥ 39.1, Moderate ≥ 33.9, Low below.
  1,097 of 33,755 change (3.2%); a handful move two tiers because the
  England overlaps are wider.

After the re-cut the ranges cannot overlap, by construction, in the
static view — and because the dashboard's dynamic re-tiering anchors to
the static assignment by rank position, the dynamic view's ranges come
out contiguous automatically too.

## What it does not change

Tier sizes are preserved exactly (London 1,265 / 616 / 1,247 / 1,866),
so the "Tier 1 contains 25.3% of London LSOAs" line and the Tier-1 KPI
count are untouched. Officer allocation never used tiers the within-BCU
split is proportional to the score so not a single LSOA's proposed
officers change. The leaderboard ranks by score and is untouched. The
dynamic layer multiplies the score and is untouched.

## The claim that has to be restated, and how

The threshold *calibration* is unaffected: it remains true that the
cut-points were learned on a 24-month fit and validated against the next
12 months' realised hotspots, AUC 0.92 London / 0.83 England. Say that
exactly as before.

The *capture* statistic needs one word of care. "Tier 1 captured 81.7% of
realised future hotspots" was measured on the calibration-time
assignment. The re-cut labels are the *deployed application* of those
thresholds to the current score, so do not quote 81.7% as a property of
the relabelled map. The honest phrasing, which is also the standard one
for any deployed risk model:

> The tier boundaries were calibrated out-of-time — fit on the first 24
> months, validated against the hotspots that actually materialised in
> the following 12 (AUC 0.92; the calibrated top tier captured 81.7% of
> them). The live tool applies those validated thresholds to the current
> 36-month score, so the label on screen is always consistent with the
> number next to it.

This is how credit scoring works: the band you see today is a pure
function of your current score, and the cut-offs were learned from
history. Nobody calls a deployed scorecard circular for applying its
validated thresholds forward. The circularity objection only bites if the
thresholds had been *chosen* on the displayed score — they weren't, and
the AUC line is the proof.

## Anticipated Q&A

**"Isn't cutting tiers on the score you display circular?"** The
validation already happened on held-out time; applying learned cuts
forward is the point of having learned them. See the credit-scoring
framing above.

**"Why not keep the calibration labels as-is?"** Because the
tool then shows equal scores with different labels (the Havering/Enfield
pair), and the only defence is a methodology lecture mid-briefing. The
calibration labels answer "what did the model flag in validation"; an
operational tool should answer "where does this place stand now, under
the validated rule".

**"Does this overwrite Natalia's Phase 6 output?"** The parquet is
untouched; the re-cut is a presentation-layer step (and 94% of London
labels are identical anyway). If the group ratifies it, the re-cut can
move upstream into the Phase 6 pipeline so the report and dashboard
agree.

## Implementation

Dashboard-side first, so both versions can be compared before the group
decides. One block in `load_clusters()` in `10_dashboard.py`
(`CBL_16/dashboard/` is the editing home, mirror to `analysis/`):

```python
# Option B re-cut: tier = rank cut on the displayed score, sizes
# preserved from Phase 6. See OPTION_B_SCORE_CUT_TIERS.md.
sizes = df["tier"].value_counts().sort_index()
order = df["risk_score_scaled"].rank(ascending=False, method="first")
cuts = sizes.cumsum()
df["tier"] = np.searchsorted(cuts.to_numpy(), order.to_numpy(),
                             side="left") + 1
```

Then: update the Method-tab caption (drop the overlap explanation, state
the boundaries instead — they become a feature, not a caveat), update
`DASHBOARD_GUIDE.md` and the tier note in `TIERING_DECISION.md`, and
redeploy the Space. If ratified, port the same re-cut into the Phase 6
pipeline in `CBL-16/` so the report tables match the dashboard.

## Decision needed

Natalia to confirm she's comfortable with the deployed-thresholds framing
and the 6% / 3.2% relabel. If yes, Guney implements and redeploys; the
report's methodology section gets the one-paragraph restatement above.
