# Final frozen 200-PR validation — 2026-09-06

This is the final full-corpus validation of commit
`b2f010964224ca09f84ea60da795cb25f7796617`. No code or benchmark labels were
changed during the run.

The machine-readable report is
`output/python-100-repo-benchmark-v1/agentic/final-frozen-b2f0109-200-pr-runtime/report.json`.
Its SHA-256 is `031675cfa6fba4635426d4b7150c79f4fe005f7f1410b1d31b27690b5b6c52d6`.

## Run integrity

- 100 public repositories and 200 cases: 100 mechanically reversed public fixes and 100 clean
  public-fix directions.
- All 200 repository roots were verified as populated Git worktrees at the expected commits before
  model calls. Their tracked-file counts ranged from 6 to 13,408.
- All 200 executions completed. Full-role success was 99%; two cases completed in degraded mode after
  one Worker failure each.
- Scanner was enabled independently and Lead revision remained optional with a one-round upper bound.
- The run used 16,540,570 tokens and 1,865 model calls.
- The run was conservatively interrupted after case 140 when five consecutive difficult risk cases
  missed. Offline comparison showed that the prior frozen run also missed all five and that the new
  run was ahead on the identical 140-case prefix, so the same checkpoint was resumed without rerunning
  completed cases.

## Final result and prior frozen comparison

| Measure | Prior frozen run | Final frozen run |
| --- | ---: | ---: |
| Strict true positives / false negatives | 54 / 46 | **57 / 43** |
| Strict recall | 54.00% | **57.00%** |
| Exact labelled-target recall | 61.00% | **64.00%** |
| Worker formal target recall | 77.00% | **85.00%** |
| Worker-published target recall | 52.00% | **54.00%** |
| Taxonomy accuracy on detected targets | 88.52% | **89.06%** |
| Clean cases without a published Finding | 94 / 100 | **95 / 100** |
| Label-relative false positives | 24 | **22** |
| Label-relative precision | 69.23% | **72.15%** |
| Label-relative F1 | 60.67% | **63.69%** |
| Scanner raw / strict true positives | 19 / 19 | 19 / 19 |
| Scanner strict publication rescues | 9 | **10** |
| Scanner unique strict true positives | 2 | 2 |
| Execution success | 200 / 200 | 200 / 200 |
| Total tokens | 16,588,299 | **16,540,570** |
| Model calls | 1,873 | **1,865** |

The full run therefore did not show a regression from allowing complete verified proof to supersede a
stale Worker-confidence estimate. Recall, clean silence, label-relative precision, token use, and call
count all improved slightly. Because model sampling is stochastic, the magnitude of these differences
must not be presented as a deterministic effect of the code change.

## Publication-rule audit

Four published findings had a Worker confidence below the old raw-confidence threshold but passed the
new complete-proof rule. One was the strict target in the AI Chat risk case. Three appeared on clean
directions: Django Haystack, PyTenable, and Tracecat. PyTenable and Tracecat had also published findings
in the prior run, while Django Haystack was new. These are the main residual publication-risk cases to
inspect if unexpected findings are later adjudicated.

The aggregate clean-direction signal did not worsen: five clean cases published a Finding versus six
in the prior run. The five current cases were Django Haystack, Calibre, PyTenable, Tracecat, and Auto
Post History. Clean-direction publication is a stronger false-positive warning, but it is not an
adjudicated real-world false positive: this targeted-label benchmark does not prove that unexpected
findings are invalid.

## Interpretation boundary

This corpus uses real public repositories, PRs, and full source trees. Its risk direction is
mechanically reversed from a public fix, so it is not a naturally occurring random risky-PR sample.
The formal readiness gate remains false because it requires at least 300 cases and complete repository
isolation across splits. The reported precision is label-relative until all unexpected findings are
manually adjudicated.
