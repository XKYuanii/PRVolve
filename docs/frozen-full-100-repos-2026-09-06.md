# Frozen 100-repository evaluation — 2026-09-06

This audit records the first complete run of the frozen agentic review pipeline at commit
`f8543925da73371d78395e9322baf4091b902aa5` (`agentic-review-freeze-2026-09-05`).
The ignored machine-readable report is
`output/python-100-repo-benchmark-v1/agentic/frozen-full-100-repos/report.json` with SHA-256
`f87ec80c9f4592c2cc31de5659b58c8e72002c600e8f1b49d1ec8532da664273`.

## Run integrity

- 100 unique public repositories and 200 complete-source worktrees
- 100 mechanically reversed public fixes (risk direction) and 100 public fixes (clean direction)
- 200/200 executions completed; execution success rate 100%
- Full-role success rate 99%; two cases completed in degraded mode after one Worker failure
- Scanner enabled independently; Lead-selected revision limit set to one round
- 16,588,299 total tokens, 1,873 model calls, and 10,014.266 seconds wall time

## Final metrics

| Measure | Result |
| --- | ---: |
| Strict true positives / false negatives | 54 / 46 |
| Strict recall | 54.00% |
| Exact labelled-target recall | 61.00% |
| Worker formal target recall | 77.00% |
| Worker-published target recall | 52.00% |
| Taxonomy accuracy on detected targets | 88.52% |
| Clean cases without a published Finding | 94 / 100 |
| Label-relative precision | 69.23% |

The precision value is not a real-world precision estimate: unexpected findings on the targeted-label
dataset have not been manually adjudicated. Six clean-direction cases did publish a Finding and are the
stronger false-positive signal.

## Publication funnel

Workers produced a formal finding at the labelled target in 77 risk cases, but only 52 survived as a
Worker-published target. Scanner findings were strict true positives in all 19 cases where they fired;
they rescued nine final strict publications, including one exact target not formally found by a Worker.
The combined final exact-target count was 61 and the combined strict count was 54.

Of the 77 Worker-formal targets, 17 remained absent from the final exact-target output even after Scanner
rescue. Fifteen of those had an inconclusive Critic verdict. One particularly important counterexample,
`python-public-stargazing-none-pollution-53-regression`, had a fully accepted Critic proof and explicit
Lead selection but was still suppressed by the model-confidence threshold. This is evidence that raw
model confidence can incorrectly override completed publication proof.

## Interpretation limits

The benchmark is based on real public fixes and full repositories, but the risk side is mechanically
reversed from those fixes rather than sampled as naturally occurring unreviewed PRs. The report's
readiness gate remains false because it requires at least 300 cases and because a small number of
repositories overlap between train, validation, and holdout splits. Aggregate results are valid for this
200-case corpus; they must not be presented as an isolated random holdout or as adjudicated real-world
precision.
