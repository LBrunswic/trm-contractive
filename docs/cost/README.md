# Cost model scripts

These scripts produce the numbers in [../COST_ESTIMATE.md](../COST_ESTIMATE.md), the dollar cost estimate for
[../CVPR_BENCHMARK_PLAN.md](../CVPR_BENCHMARK_PLAN.md), apart from a few sums or ratios of printed values and one
rough estimate flagged in its §9. They need Python 3 and the standard library only, and each runs in under a second.
Run them from the repository root:

```bash
python3 docs/cost/dollar_cost_model.py   # scopes, scenarios S1-S5, effective $ per G-h, per-table $, top rows
python3 docs/cost/dollar_cost_extra.py   # $ per G-h for every offer, break-even, package, tranches, cuts, options
python3 docs/cost/roofline.py            # per-kernel roofline behind the throughput factors S
```

| File | What it holds | Sections of `COST_ESTIMATE.md` it prints |
|---|---|---|
| `dollar_cost_model.py` | The plan's rows (lo/hi G-h per row, copied from the plan's tables), the L-core upgrade rows, the scenario parameters (prices, throughput S, utilisation U, failure allowance, disk, egress, VAT) and the cost formula of §1.3 | §3 (scope sizes and work including failures), §5 (scenario table, effective $ per lo-G-h), §7 (per-table breakdown), §8 (top rows) |
| `dollar_cost_extra.py` | Imports the model; reads the offer list from `prices_2026-10-27.csv` | §0, §2 (every offer, sorted by central $ per G-h; the document's table shows a selection), §4 (overheads, continuous rental), §5 (wall-clock), §6 (package, tranches, cuts, optional rows, node gate), §8 (per-arm sums), §9 |
| `roofline.py` | Assumed achieved TFLOP/s and bandwidth per GPU (spec-derived), calibrated to a fast RTX 5080 | §1.2 (memory-bound share on the 5080; roofline speed-ups that bound the S ranges) |
| `prices_2026-10-27.csv` | One row per cloud offer used: GPU, provider, offer, tier (on-demand or interruptible), $ per GPU-hour, throughput key, date, public source URL | §2 |

**Inputs.** All inputs are constants in the scripts or rows of the CSV, as of the 27 Oct 2026 snapshot. Vast.ai
prices in the CSV are aggregates (medians, or min-bid medians for interruptible offers) over the live offer search at
22:30–22:40 UTC; the individual marketplace listings are not included. The tranche split in `dollar_cost_extra.py`
(share of the Tier 0 and L-core cost falling in Tranche 2: 81/359 central, 161/708 pessimistic) is a fixed input
from a row-by-phase assignment of the plan's schedule.

**Rerunning with new data.** After the G0 probe (`COST_ESTIMATE.md` §6.5), replace the assumed throughput `S` and
utilisation `U` of the scenario in `SC` (in `dollar_cost_model.py`; per-offer `S` ranges are in `dollar_cost_extra.py`)
with measured values. For new prices, edit the scenario prices `p` in `SC` and the rows of the CSV (or add a CSV with a
new date and point `_CSV` in `dollar_cost_extra.py` to it).
