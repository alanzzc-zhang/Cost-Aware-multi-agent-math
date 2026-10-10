# Historical results and reproducibility

Source: [original final report](final_report.pdf), Tables 4–6. These values were transcribed from the report, not recalculated from raw logs or rerun for this release.

## Main comparison — 150 questions (Table 4)

| Metric | Baseline | Advanced, original prompts |
| --- | ---: | ---: |
| Questions | 150 | 150 |
| Total tokens | 785,042 | 1,237,331 |
| Total runtime (s) | 4,731.75 | 3,775.56 |
| Mean runtime (s/question) | 31.54 | 25.17 |
| Reported accuracy | 30% | 36% |
| Approximate total API cost | $5 | $1 |

Baseline used GPT-4.1; advanced used a GPT-4.1-mini/GPT-4.1 mix. The estimated dollar saving is approximately 80%, while token usage **increases**. Model choice and routing are confounded. Cost estimates use historical prices and are not a current price quotation.

## Advanced prompt revision — 150 questions (Table 5)

| Metric | Original | Revised |
| --- | ---: | ---: |
| Total tokens | 1,237,331 | 1,164,019 |
| Total runtime (s) | 3,775.56 | 3,817.55 |
| Mean runtime (s/question) | 25.17 | 25.45 |
| Re-evaluated accuracy | 36.0% | 38.67% |

The 5.9% token reduction compares two **advanced** prompt versions, not advanced against baseline. Revised-run prompt tokens were 950,168. Both historical versions are not available; the supplied YAML files cannot be definitively mapped to a historical run.

## Text-only check — 100 questions (Table 6)

| Metric | Baseline | Advanced |
| --- | ---: | ---: |
| Total tokens | 454,680 | 710,374 |
| Total runtime (s) | 2,369.40 | 2,576.63 |
| Mean runtime (s/question) | 23.69 | 25.76 |
| Re-evaluated accuracy | 41.0% | 47.0% |

The advanced system improves reported accuracy here but is slower and uses more tokens. The post-hoc scorer normalized formats such as fractions, roots, and exponents; that scorer is not included.

## Required to reproduce these tables

- `OE_MM_frontier_format_renumbered.json` (150 questions), with original item IDs and preprocessing provenance.
- `OE_TO_textonly_100_frontier_format.json` (100 questions).
- Per-question outputs and token logs for each run.
- The common post-hoc answer evaluator and its exact settings.
- Original and revised advanced prompts, tied to run configurations.
- Model versions, full commands, run dates, and cost assumptions.

The current CLI scorers differ: baseline uses its own normalization; advanced uses string equality after stripping whitespace. Neither should be assumed to reproduce the post-hoc report numbers.

## Code snapshot caveats

Core Python and YAML files were preserved unchanged for this release. In both pipelines, Critic/Judge outputs do not replace the Solver answer used for scoring. In the advanced PLANNER route, the summary output and generated plan are not used to choose the final answer; model list and run count are used. The report discusses the intended architecture more broadly than this snapshot implements.

Advanced `solver_calls` counts stored solver-result objects: DIRECT stores the A-base answer as one such object even though no separate Solver API call occurred. Do not interpret that field as a literal count of separate Solver requests for DIRECT. Advanced token totals are not written by its main output path.

The main evaluation includes diagram-dependent questions without image inputs. Small sample size, answer formatting, unverified data keys, and missing visual context limit interpretation. These results are exploratory rather than evidence of general superiority.
