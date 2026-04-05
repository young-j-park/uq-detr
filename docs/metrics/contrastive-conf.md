# ContrastiveConf (Image-Level Reliability)

**`uq_detr.contrastive_conf(all_queries, ...)`**

ContrastiveConf quantifies **per-image reliability** by contrasting the confidence of positive and negative predictions. Higher scores indicate more reliable images.

## Motivation

Our theoretical analysis (Proposition 3) shows that when a DETR model is uncertain about an image, the confidence gap between primary (optimal positive) and secondary (optimal negative) predictions shrinks. ContrastiveConf exploits this by measuring the gap as a reliability signal.

## Definition

$$
\text{ContrastiveConf}(x) = \text{Conf}^+(x) - \lambda \cdot \text{Conf}^-(x)
$$

where:

- $\text{Conf}^+(x)$ = average max-foreground-confidence of positive predictions
- $\text{Conf}^-(x)$ = average max-foreground-confidence of negative predictions
- $\lambda$ = scaling factor (recommended: 5.0--10.0)

The positive/negative split is determined by a post-processing method (e.g., confidence threshold).

## Usage

```python
import uq_detr

# all_queries: list of Detections, one per image (full DETR output, before post-processing)
scores = uq_detr.contrastive_conf(
    all_queries,
    method="threshold",   # post-processing to split pos/neg
    param=0.3,            # threshold value
    lambda_=5.0,          # scaling factor
)
# scores: np.ndarray of shape (N_images,)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `all_queries` | `list[Detections]` | required | Full query set (all queries) per image |
| `method` | `str` | `"threshold"` | Post-processing method: `"threshold"` or `"topk"` |
| `param` | `float` | `0.3` | Method parameter (threshold value or k) |
| `lambda_` | `float` | `5.0` | Scaling factor for negative confidence |

## Fitting Lambda

**`uq_detr.fit_lambda(all_queries, reliability, ...)`**

Use `fit_lambda` to find the optimal $\lambda$ on a validation set. It sweeps over candidate values and returns the one that maximizes the Pearson correlation with a user-provided per-image reliability measure.

```python
import uq_detr

# Step 1: Fit lambda on validation set
best_lam, best_pcc = uq_detr.fit_lambda(
    val_queries,          # full DETR queries on val set
    val_per_image_ap,     # per-image reliability (e.g., per-image AP)
    method="threshold",
    param=0.3,
)
print(f"Best lambda: {best_lam} (PCC={best_pcc:.4f})")

# Step 2: Apply on test set
test_scores = uq_detr.contrastive_conf(
    test_queries,
    method="threshold",
    param=0.3,
    lambda_=best_lam,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `all_queries` | `list[Detections]` | required | Full query set per image (validation) |
| `reliability` | `np.ndarray` | required | Per-image reliability scores, shape `(N,)`. Can be per-image AP, negated per-image OCE, or any scalar measure. |
| `method` | `str` | `"threshold"` | Post-processing method |
| `param` | `float` | `0.3` | Post-processing parameter |
| `lambda_range` | `list[float]` | `[0, 0.25, ..., 10, 15, 20]` | Candidate lambda values to search |

!!! tip
    The optimal $\lambda$ typically lies in the range 5.0--10.0 (see paper, Figure 12). Excessively large $\lambda$ degrades performance, especially on out-of-distribution data.

## Selecting the Post-Processing

The paper recommends using OCE to find the best post-processing configuration on a validation set:

$$
A^* = \arg\min_A \text{OCE}(A \circ \hat{Y}_\theta(X_\text{val}),\; Y_\text{val})
$$

Then use this $A^*$ for both `contrastive_conf` and `fit_lambda`.

## Correlation with Image-Level AP

ContrastiveConf consistently achieves higher Pearson correlation with per-image AP than using $\text{Conf}^+$ alone, across in-distribution and out-of-distribution settings (see paper, Table 2).
