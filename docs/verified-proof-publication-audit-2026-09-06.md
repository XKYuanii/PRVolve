# Verified-proof publication audit — 2026-09-06

Commit `37ebb1d` removes the raw Worker-confidence veto only when all of the following are true:

- the Lead selected the finding;
- the Critic or grounded Lead review completed all six publication obligations;
- the proof is backed by repository evidence;
- the existing location, evidence, and release gates still pass.

Bare `publication_ready` flags, inconclusive reviews, unselected findings, missing repository evidence,
invalid locations, and missing fix/test guidance remain blocked. The implementation records
`verified_proof_supersedes_confidence` on the finding so the final confidence gate can distinguish a
completed proof from an unverified low-confidence claim.

## Deterministic replay

The prior 100-repository report was replayed without new model calls. Under identical stored Worker,
Critic, Lead, and evidence outputs:

- `python-public-stargazing-none-pollution-53-regression` changed from rejected to confirmed because its
  Critic proof was accepted, every premise was verified, the Lead selected it, and its only remaining
  blocker was confidence 0.50;
- the other 16 formal-target findings absent from the prior final output remained unpublished because
  they did not contain a completed publication proof.

This replay isolates the code change from model sampling: one intended recovery and no additional
release among the 17 historical cases.

## Fresh targeted run

The ignored machine-readable report is
`output/python-100-repo-benchmark-v1/agentic/verified-proof-17-plus-8-clean/report.json`.
Its SHA-256 is `a117408ca2835f6537a27f881e1593251711e68287cf356c244a3d3c69ba0dc2`.
It contains all 17 historical formal-target publication losses plus eight clean cases sampled uniformly
from the prior 94 silent clean cases with Python `random.Random(20260906)`.

| Measure | Result |
| --- | ---: |
| Historical risk cases | 17 |
| Strict true positives | 8 / 17 |
| Exact labelled-target detections | 9 / 17 |
| Worker formal target detections | 15 / 17 |
| Worker-published targets | 9 / 17 |
| Random clean cases without a Finding | 8 / 8 |
| Scanner findings | 0 |
| Executions completed | 25 / 25 |

The run took 1,187.985 seconds, made 230 model calls, and used 2,164,347 total tokens.

The fresh run published five label-relative false positives, all on risk cases: one extra Sonar finding,
one off-target Mnemo finding, and three Lenie findings (one at the labelled target but with a mismatched
taxonomy). None had low confidence; all were 0.8 or 0.99, so the new confidence exception did not cause
their publication.

The fresh run is stochastic and therefore cannot attribute its eight strict recoveries to the code
change. In particular, Stargazing did not reproduce its prior accepted proof: this time its Critic proof
was incomplete and the Lead did not select it, so it correctly remained blocked. The deterministic replay
above is the controlled evidence for the new rule; the fresh run is evidence that the rule did not create
an obvious clean-case release regression in the eight sampled controls.

All 223 unit tests passed before this run.
