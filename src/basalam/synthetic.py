"""Synthetic data generator mimicking the BaSalam products schema.

Used by the smoke pipeline and CI so that no marketplace records are needed
to run or test the code end-to-end. The generator reproduces the three
structurally important properties of the real data:

1. Extreme zero inflation of the weekly sales target.
2. Seller (vendor) grouping with skewed vendor sizes.
3. The `_score` column: a deterministic function of the target for nonzero
   rows (500*(y+1)/(y+2)) — the leakage trap that the audit must catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic(
    n_products: int = 20_000,
    n_vendors: int = 400,
    n_categories: int = 40,
    seed: int = 42,
    target_nonzero_share: float = 0.032,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Skewed vendor sizes (Pareto), as in the real marketplace
    vendor_labels = np.array([f"v{i}" for i in range(n_vendors)])
    weights = rng.pareto(1.6, n_vendors) + 1.0
    vendor = rng.choice(vendor_labels, size=n_products, p=weights / weights.sum())

    category_labels = np.array([f"c{j:02d}" for j in range(n_categories)])
    category = rng.choice(category_labels, size=n_products)

    vendor_quality_map = dict(zip(vendor_labels, rng.normal(0, 1.0, n_vendors)))
    quality = np.array([vendor_quality_map[v] for v in vendor])

    # log-price shifts +0.3*vendor quality: the fold-internal vendor mean price
    # is therefore an informative (but noisy) vendor-quality proxy.
    price = np.exp(11.5 + 0.7 * rng.standard_normal(n_products) + 0.3 * quality)
    primary_price = price * rng.uniform(1.0, 2.5, n_products)
    rating_average = np.clip(rng.beta(4, 1.2, n_products) * 5, 0, 5)
    rating_count = rng.poisson(3, n_products)
    stock = rng.poisson(50, n_products)
    preparation_days = np.clip(rng.poisson(2, n_products), 0, 30)
    vendor_score = np.clip(
        0.5 + 0.02 * quality + 0.15 * rng.standard_normal(n_products), 0, 1
    )

    # Latent sale probability: vendor quality dominates and is only weakly
    # visible in per-product features (vendor_score is noisy). The exploitable
    # vendor signal lives in fold-internal aggregates (e.g. vendor mean price),
    # which is exactly the mechanism the grouped-vs-random gap (H1) is about.
    logit = -5.8 + 2.5 * quality + 1.0 * (rating_average / 5 - 0.7) - 0.4 * (np.log(price) - 11.5 - 0.3 * quality)
    p_sale = 1.0 / (1.0 + np.exp(-logit))
    sold = rng.random(n_products) < p_sale

    y = np.zeros(n_products, dtype=np.int64)
    n_sold = int(sold.sum())
    y[sold] = 1 + rng.poisson(0.7, n_sold)

    # _score reproduces the real leakage: deterministic for nonzero rows
    score = rng.uniform(0, 300, n_products)
    nz = y > 0
    score[nz] = 500.0 * (y[nz] + 1) / (y[nz] + 2)

    df = pd.DataFrame(
        {
            "_id": [f"p{i}" for i in range(n_products)],
            "_score": score,
            "sales_count_week": y,
            "price": price,
            "primaryPrice": primary_price,
            "rating_average": rating_average,
            "rating_count": rating_count,
            "stock": stock,
            "preparationDays": preparation_days,
            "vendor_id": vendor,
            "vendor_score": vendor_score,
            "categoryId": pd.factorize(category)[0],
            "new_categoryId": pd.factorize(category)[0],
            "category": category,
        }
    )
    return df
