# Experiment Results

This directory contains saved evaluation metrics, diagnostic tables, and figures. Generated artifacts are excluded from Git by default; only selected, verified outputs should be committed.

## Official-split evaluation

The primary experiment uses the official EuroSAT RGB splits:

- Gallery: 16,200 training images.
- Validation queries: 5,400 validation images.
- Test queries: 5,400 test images.

The validation split selects the reranking parameters. The test split is evaluated after selection.

### Main results

- `metrics/official_split_reranking.json`: experiment configuration, selected parameters, test metrics, and differences from the cosine baseline.
- `metrics/official_split_reranking.csv`: tabular comparison of test metrics.
- `metrics/paired_bootstrap_test.json`: paired-bootstrap settings and confidence intervals.
- `metrics/paired_bootstrap_test.csv`: method-minus-baseline differences and confidence intervals.
- `metrics/baseline_metrics.json`: cosine-baseline metrics, including gallery and query counts.

### Selective-reranking experiment

- `metrics/selective_split_reranking.json`: validation-selected selective-reranking configuration, validation metrics, test metrics, and reranked-query count.
- `metrics/selective_split_reranking.csv`: tabular comparison of cosine retrieval, fixed reranking, adaptive reranking, and selective reranking.

The selective experiment uses the same official gallery, validation queries, and test queries as the primary experiment. It selects its margin threshold using validation mAP@100 and evaluates the selected configuration on the held-out test split.

### Figures and diagnostics

- `figures/reranking_comparison.png`: three-way comparison of the cosine baseline, fixed reranking, and adaptive reranking on the official test split.
- `figures/class_metrics.png`: baseline metrics broken down by test class.
- `figures/top1_confusion_matrix.png`: baseline top-1 class confusion matrix on the test split.
- `figures/failure_examples.png`: selected baseline retrieval failures.
- `metrics/class_metrics.csv`: per-class baseline metrics.
- `metrics/top1_confusion_matrix.csv`: confusion matrix counts.
- `metrics/retrieval_failures.csv`: metadata for the selected failure examples.

The class-level and failure-analysis artifacts describe the cosine baseline, not the reranked methods. The existing comparison figure and paired-bootstrap artifacts cover the original cosine, fixed, and adaptive methods; they do not include a separate selective-method bootstrap analysis.

## Interpretation

Both fixed and adaptive reciprocal reranking improve mAP@100 but reduce HitRate@5 and HitRate@10. The observed Top-1 differences are inconclusive under the reported paired-bootstrap intervals.

The intervals quantify query-sampling uncertainty conditional on the selected gallery, encoder, dataset split, and validation-selected settings. They do not establish generalization to other datasets or encoders.

### Selective-reranking findings

The additional selective experiment selected a margin threshold of 0.20 using validation mAP@100. This threshold reranked all 5,400 test queries.

Selective reranking produced the same reported test metrics as fixed reranking with alpha = 0.50:

| Method | Top-1 | Hit@5 | Hit@10 | mAP@100 |
|---|---:|---:|---:|---:|
| Cosine baseline | 0.8789 | 0.9707 | 0.9843 | 0.6786 |
| Fixed reciprocal reranking | 0.8724 | 0.9507 | 0.9683 | 0.6905 |
| Adaptive reciprocal reranking | 0.8811 | 0.9589 | 0.9726 | 0.6884 |
| Selective reciprocal reranking | 0.8724 | 0.9507 | 0.9683 | 0.6905 |

The selective experiment did not demonstrate a reduction in reranking workload or an advantage over fixed reranking. This result does not establish that selective reranking in general is ineffective; it shows that this particular threshold-selection heuristic did not deliver its intended benefit under the evaluated configuration.

The result should be interpreted as conditional on the dataset, gallery, encoder, split, and tested threshold candidates. No claim of methodological novelty or generalization is made.

## Exploratory results

The following files belong to an earlier exploratory evaluation with 4,000 gallery images and 1,000 queries:

- `metrics/reranking_comparison.json`
- `metrics/reranking_comparison.csv`
- `metrics/adaptive_query_weights.csv`

These files must not be interpreted as outputs from the official-split evaluation. Their parameter settings and results differ from the primary experiment.

## Reproduction

Use the commands in the project-level README to regenerate the baseline, reranking evaluation, selective-reranking evaluation, bootstrap analysis, figures, and failure diagnostics.

Generated outputs can vary with hardware, software versions, dataset revisions, and pretrained weights.
