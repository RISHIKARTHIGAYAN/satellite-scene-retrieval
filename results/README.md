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

### Figures and diagnostics

- `figures/reranking_comparison.png`: three-way comparison of the cosine baseline, fixed reranking, and adaptive reranking on the official test split.
- `figures/class_metrics.png`: baseline metrics broken down by test class.
- `figures/top1_confusion_matrix.png`: baseline top-1 class confusion matrix on the test split.
- `figures/failure_examples.png`: selected baseline retrieval failures.
- `metrics/class_metrics.csv`: per-class baseline metrics.
- `metrics/top1_confusion_matrix.csv`: confusion matrix counts.
- `metrics/retrieval_failures.csv`: metadata for the selected failure examples.

The class-level and failure-analysis artifacts describe the cosine baseline, not the reranked methods.

## Interpretation

Both fixed and adaptive reciprocal reranking improve mAP@100 but reduce HitRate@5 and HitRate@10. The observed Top-1 differences are inconclusive under the reported paired-bootstrap intervals.

The intervals quantify query-sampling uncertainty conditional on the selected gallery, encoder, dataset split, and validation-selected settings. They do not establish generalization to other datasets or encoders.

## Exploratory results

The following files belong to an earlier exploratory evaluation with 4,000 gallery images and 1,000 queries:

- `metrics/reranking_comparison.json`
- `metrics/reranking_comparison.csv`
- `metrics/adaptive_query_weights.csv`

These files must not be interpreted as outputs from the official-split evaluation. Their parameter settings and results differ from the primary experiment.

## Reproduction

Use the commands in the project-level README to regenerate the baseline, reranking evaluation, bootstrap analysis, figures, and failure diagnostics.

Generated outputs can vary with hardware, software versions, dataset revisions, and pretrained weights.