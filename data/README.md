# Dataset notes

| File | Records | Status |
| --- | ---: | --- |
| `smoke_test.json` | 3 | New elementary examples added for functional checks; answers checked manually |
| `custom_math_200.json` | 200 | Supplied with the project; provenance and answer quality not independently established |
| `frontiermath_10.json` | 10 | Supplied with the project; filename does not establish affiliation with or origin from the official FrontierMath benchmark |

The two supplied files are preserved byte-for-byte. They are **not** the report's 150-question OlympiadBench set or its supplementary 100-question text-only set. Do not present runs on them as reproductions of the report.

The supplied answer keys have not been fully audited. Check their correctness before treating accuracy on these examples as a benchmark result.

Input format: a JSON array of objects containing `id`, `question`, and optionally `answer` and `difficulty`. The routing decision is derived from A-base's output, not the stored difficulty label.

The original benchmark source is identified in the report as OpenBMB/OlympiadBench. The processed subsets, item mapping, and preprocessing script are not supplied here. Document source and redistribution terms when adding them. No third-party dataset license is assigned by this repository.
