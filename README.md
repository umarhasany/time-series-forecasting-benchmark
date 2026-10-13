# Benchmarking Statistical, Machine-Learning, and Foundation Models for Time-Series Forecasting

This project compares three forecasting paradigms across two daily retail-demand datasets:

- **ETS** - statistical forecasting
- **LightGBM** - global machine-learning forecasting
- **Chronos-2** - pretrained time-series foundation model

The goal is to study whether model rankings observed on a relatively simple public benchmark transfer to a richer, more industrial-style forecasting setting, and how the models differ in **accuracy, uncertainty, and computational cost**.

## Research Questions

1. Do statistical, machine-learning, and foundation-model rankings remain stable across datasets?
2. Does a pretrained time-series foundation model provide a clear advantage over classical and machine-learning approaches?
3. How do the models compare when uncertainty calibration and computational cost are considered alongside point-forecast accuracy?
4. Do additional external covariates such as price, events, and SNAP indicators consistently improve machine-learning forecasts?

## Datasets

### 1. Kaggle Store Item Demand Forecasting

The first dataset is the Kaggle Store Item Demand Forecasting Challenge dataset.

- 10 stores
- 50 items
- 500 store-item time series
- Daily sales
- 2013-01-01 to 2017-12-31
- 913,000 observations

This dataset provides a relatively clean and regular retail-demand benchmark.

### 2. M5 Forecasting - Accuracy

The second dataset is the public M5 Forecasting dataset derived from Walmart retail sales.

The original dataset contains:

- 30,490 item-store time series
- 3,049 items
- 10 stores
- 3 states
- 3 product categories
- 7 departments
- Daily sales
- Calendar, event, SNAP, and price information

For this project, a reproducible sample of **50 series per store** was selected using a fixed random seed, giving **500 M5 series**. This keeps the number of series balanced with the first dataset while preserving a substantially more intermittent and heterogeneous demand environment.

M5 is used here as a **public industrial-style proxy**, not as private industrial data.

## Experimental Protocol

Both datasets use the same high-level forecasting protocol:

- **500 time series**
- **Daily forecasting**
- **28-day validation window**
- **28-day final test horizon**
- Chronological splits only
- No random train-test shuffling

The validation period is used for configuration selection. The final 28 days are kept untouched for final testing.

### Accuracy Metrics

Three complementary point-forecast metrics are reported:

- **sMAPE** - symmetric mean absolute percentage error
- **WAPE** - weighted absolute percentage error
- **RMSE** - root mean squared error

Using several metrics is important because M5 contains many zero and very low-demand observations, which makes percentage-based metrics particularly sensitive.

### Uncertainty Evaluation

All models are evaluated against a target **80% prediction interval**.

- **ETS** - validation-calibrated conformal residual interval
- **LightGBM** - validation-calibrated conformal residual interval
- **Chronos-2** - native 0.1 and 0.9 forecast quantiles

For each model, the project reports:

- Observed interval coverage
- Mean prediction-interval width

The interval-generation mechanisms differ, so coverage and width should be interpreted as practical forecasting characteristics rather than as identical probabilistic constructions.

### Computational Cost

The benchmark also records:

- Fitting or task-specific training time
- Inference time

Chronos-2 is used zero-shot, so its task-specific fitting time is reported as zero. This does **not** mean that Chronos-2 itself was never trained; it is a pretrained foundation model.

## Models

### ETS

ETS is used as the statistical forecasting approach.

Each time series receives its own exponential-smoothing model with:

- additive trend
- additive weekly seasonality
- seasonal period = 7

Validation compares additive and damped-trend configurations.

### LightGBM

LightGBM is used as a global machine-learning forecaster trained across all 500 series.

Features include:

- store and item identifiers
- day of week
- month
- day of year
- lag 1
- lag 7
- lag 14
- lag 28
- 7-day rolling mean
- 28-day rolling mean

Forecasting is recursive over the 28-day horizon, meaning earlier predictions become part of the history used to generate later lag features.

### Chronos-2

Chronos-2 is a pretrained Transformer-based time-series foundation model.

It is evaluated:

- zero-shot
- without task-specific fine-tuning
- with a 28-day prediction horizon
- using native quantile forecasts for uncertainty

## Final Accuracy Results

Lower is better for all three metrics.

| Dataset | Model | Category | sMAPE | WAPE | RMSE |
|---|---|---|---:|---:|---:|
| Store-Item | ETS | Statistical | 23.56% | 23.20% | 12.56 |
| Store-Item | LightGBM | Machine Learning | 13.99% | 12.39% | 7.16 |
| Store-Item | **Chronos-2** | Foundation Model | **13.77%** | **12.07%** | **6.94** |
| M5 | **ETS** | Statistical | **139.46%** | 75.59% | 2.28 |
| M5 | LightGBM | Machine Learning | 141.22% | 78.85% | **2.06** |
| M5 | Chronos-2 | Foundation Model | 162.30% | **67.90%** | 2.08 |

### Store-Item Ranking

All three accuracy metrics produce the same ranking:

1. Chronos-2
2. LightGBM
3. ETS

Chronos-2 achieves the best point accuracy, but LightGBM performs very similarly.

### M5 Ranking

The ranking becomes metric-dependent:

**sMAPE**
1. ETS
2. LightGBM
3. Chronos-2

**WAPE**
1. Chronos-2
2. ETS
3. LightGBM

**RMSE**
1. LightGBM
2. Chronos-2
3. ETS

This is a central result of the project: **a model ranking observed on the simpler dataset does not transfer cleanly to the richer M5 setting**.

## Uncertainty Results

Target coverage is 80%.

| Dataset | Model | Interval Method | Observed Coverage | Mean Interval Width |
|---|---|---|---:|---:|
| Store-Item | ETS | Conformal residual | 65.25% | 23.90 |
| Store-Item | LightGBM | Conformal residual | 85.20% | 20.07 |
| Store-Item | **Chronos-2** | Native quantiles | **81.04%** | **17.54** |
| M5 | **ETS** | Conformal residual | **80.27%** | **1.98** |
| M5 | LightGBM | Conformal residual | 80.80% | 2.10 |
| M5 | Chronos-2 | Native quantiles | 78.64% | 2.65 |

### Interpretation

On the Store-Item dataset:

- Chronos-2 is closest to the 80% target while also producing the narrowest intervals.
- LightGBM is conservative, with 85.20% coverage.
- ETS substantially undercovers at 65.25%.

On M5:

- All three models are reasonably close to the target.
- ETS achieves 80.27% coverage with the narrowest interval.
- LightGBM achieves 80.80% coverage.
- Chronos-2 slightly undercovers at 78.64%.

Interval widths should only be compared **within the same dataset**, because the two datasets operate at different sales scales.

## Computational Cost

| Dataset | Model | Fit / Task-Specific Training Time | Inference Time |
|---|---|---:|---:|
| Store-Item | ETS | 262.80 s | 21.46 s |
| Store-Item | LightGBM | 21.91 s | **0.60 s** |
| Store-Item | Chronos-2 | 0 s local fitting | 7.15 s |
| M5 | ETS | 215.50 s | 20.44 s |
| M5 | LightGBM | 15.64 s | **0.50 s** |
| M5 | Chronos-2 | 0 s local fitting | 5.99 s |

LightGBM is the fastest inference approach by a large margin on both datasets.

ETS has the highest fitting and inference cost because a separate model is fitted for each series.

Chronos-2 requires no task-specific training in this benchmark, but inference is more expensive than LightGBM and was run with GPU support.

## External Covariate Experiment

A separate M5 experiment tests whether adding external variables improves LightGBM.

The baseline model uses historical sales, lag features, rolling statistics, identifiers, and calendar-derived features.

The covariate model additionally includes:

- sell price
- event indicator
- state-specific SNAP indicator

| M5 LightGBM Variant | sMAPE | WAPE | RMSE |
|---|---:|---:|---:|
| Historical features | 141.22% | **78.85%** | **2.06** |
| + Price, Event, SNAP | **141.03%** | 80.95% | 2.11 |

The additional covariates produce a very small improvement in sMAPE but worsen WAPE and RMSE.

Therefore, this experiment does **not** show a consistent overall benefit from the added external variables.

This illustrates an important forecasting lesson: **more available information does not automatically lead to better forecasts**. Its value depends on representation, model design, horizon, and the underlying demand process.

## Key Findings

### 1. Model rankings do not transfer cleanly across datasets

Chronos-2 ranks first on all three point-forecast metrics on the simpler Store-Item dataset.

On M5, however, no model dominates:

- ETS wins sMAPE
- Chronos-2 wins WAPE
- LightGBM wins RMSE

A leaderboard observed on one benchmark therefore cannot be assumed to generalize to another operational setting.

### 2. Foundation models are competitive, but not universally superior

Chronos-2 performs best on the first dataset and provides well-calibrated uncertainty without task-specific fitting.

On M5, however, its advantage disappears and the preferred model depends on the metric.

### 3. LightGBM provides a strong accuracy-cost trade-off

LightGBM performs close to Chronos-2 on the first dataset and achieves the lowest RMSE on M5.

It also has the fastest inference time on both datasets, making it attractive when deployment latency and computational cost matter.

### 4. Classical statistical methods can remain competitive

ETS performs poorly relative to the other models on the first dataset, but achieves the best M5 sMAPE and strong uncertainty calibration.

This reinforces the value of benchmarking classical methods rather than assuming newer model families will always dominate.

### 5. Uncertainty behavior is dataset-dependent

Chronos-2 gives the strongest uncertainty result on the Store-Item dataset.

On M5, ETS and LightGBM provide coverage closer to the target with narrower intervals.

Forecast uncertainty therefore needs to be evaluated empirically rather than inferred from model family alone.

### 6. Additional covariates did not guarantee improvement

Adding price, event, and SNAP features to LightGBM did not consistently improve the final M5 forecast.

The effect of external information should therefore be validated rather than assumed.

## Overall Conclusion

The benchmark shows that there is no universally best forecasting approach across all datasets, metrics, uncertainty requirements, and computational constraints.

The strongest model on a simple benchmark may not remain the strongest when the data becomes more intermittent, heterogeneous, hierarchical, or operationally complex.

For industrial forecasting, model selection should therefore consider more than a single leaderboard score. A practical evaluation should jointly consider:

- dataset characteristics
- forecast-error metric
- uncertainty calibration
- inference cost
- fitting cost
- deployment constraints
- availability and usefulness of external covariates

This project demonstrates a reproducible framework for carrying out that comparison across statistical, machine-learning, and pretrained foundation forecasting models.

## Repository Structure

```text
time-series-forecasting-benchmark/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── forecasting_benchmark.ipynb
├── results/
│   ├── accuracy_results.csv
│   ├── uncertainty_results.csv
│   ├── computational_cost.csv
│   └── covariate_results.csv
└── figures/
    └── ...
```

## Reproducing the Project

### 1. Clone the repository

```bash
git clone https://github.com/umarhasany/time-series-forecasting-benchmark.git
cd time-series-forecasting-benchmark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

A GPU is recommended for Chronos-2 inference but is not required for ETS or the default LightGBM experiments.

### 3. Download the datasets

The datasets are intentionally **not stored in this repository**.

Download:

- Kaggle Store Item Demand Forecasting Challenge
- Kaggle M5 Forecasting - Accuracy

Expected source files include:

```text
train.csv
sales_train_validation.csv
calendar.csv
sell_prices.csv
```

Place them in a local `data/` directory before running the notebook.

### 4. Run the notebook

Open:

```text
notebooks/forecasting_benchmark.ipynb
```

The notebook contains data preparation, validation/test splitting, model training or inference, uncertainty evaluation, covariate experiments, and final comparisons.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Statsmodels
- LightGBM
- PyTorch
- Chronos / chronos-forecasting
- Matplotlib
- Google Colab

## Notes on Reproducibility

- M5 sampling uses `random_state=42`.
- The M5 experiment samples 50 item-store series per store.
- All train-validation-test splits are chronological.
- Validation and test horizons are both 28 days.
- Forecasts are clipped at zero because negative unit-sales forecasts are not meaningful.
- Chronos-2 is evaluated zero-shot.
- M5 is treated as a public industrial-style benchmark, not as private industrial data.

## Author

**Syed Umar Hasany**

M2 - Data and Intelligence for Smart Systems  
Université Claude Bernard Lyon 1

- GitHub: https://github.com/umarhasany
- LinkedIn: https://www.linkedin.com/in/uhasany/

## License

This repository is intended for educational, research, and portfolio use. Dataset usage remains subject to the terms of the original dataset providers.
