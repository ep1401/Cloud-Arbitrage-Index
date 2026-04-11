# Cloud Arbitrage Index (CAI)

A forward-looking risk metric for AWS Spot Instances.

## Overview

Cloud Arbitrage Index (CAI) is a research project and modeling pipeline for estimating the near-term interruption risk of AWS Spot Instance pools. Rather than relying on coarse provider summaries such as long-window interruption buckets or opaque placement scores, CAI builds its own probe dataset and produces horizon-specific probability estimates that can support real scheduling and pool-selection decisions.

This repository contains the code used for three connected parts of the project:

1. **Active probing** of AWS Spot Instance pools across regions and instance types.
2. **Risk modeling** to estimate whether a pool is likely to experience interruption in a fixed future window.
3. **Policy evaluation** to test whether those predictions improve downstream cloud-allocation decisions.

The accompanying paper, **"Cloud Arbitrage Index (CAI): A Forward-Looking Risk Metric for Spot Instances,"** describes the research motivation, modeling decisions, evaluation framework, and empirical findings behind the code in this repository.

## Research question

The central question behind CAI is simple:

> If a user plans to launch a Spot Instance in a given AWS pool within the next few hours, what is the probability that it will be interrupted before a one-hour job completes?

AWS exposes useful but limited signals, such as historical interruption ranges and spot price traces, but it does not provide a direct, forward-looking, horizon-specific estimate of termination risk. CAI is designed to fill that gap.

## What the project does

At a high level, the project works as follows:

### 1. Probe the market directly

Because AWS does not expose the termination logs needed for supervised modeling, CAI collects its own data by repeatedly launching Spot Instances and observing whether they survive a fixed one-hour window. The probing setup used in the paper focuses on six AWS pools formed by:

- **Regions:** `us-east-1`, `us-west-2`
- **Instance types:** `t3a.large`, `m6a.large`, `c6i.large`

Each successful launch produces a one-hour probe outcome:

- `interrupted = 1` if the instance is reclaimed during the hour
- `interrupted = 0` if it survives the full hour and is then terminated by the scheduler

Launches that never reach the running state are logged separately but excluded from the main modeling dataset.

### 2. Compare probing strategies

The repository includes code for two probing policies:

- **Uniform baseline policy:** guarantees broad and steady coverage across pools
- **Adaptive bandit policy:** allocates extra probes toward pools with greater uncertainty

This makes it possible to compare whether smarter data collection improves downstream prediction quality under the same overall budget.

### 3. Build a forward-looking target

The model does not try to predict a single raw interruption event in isolation. Instead, it constructs a fixed-horizon target over the next **six hours**. For each pool and decision time, the pipeline aggregates future one-hour slices inside that horizon and estimates the empirical interruption rate in the upcoming block.

This fixed-window formulation was chosen because it is more stable and data-efficient than trying to fit a full survival model from sparse, highly censored spot interruption data.

### 4. Estimate risk with a baseline-plus-ML pipeline

The final CAI pipeline combines:

- a **pool-level statistical baseline** based on recent interruption behavior
- a **machine learning correction model** that uses time, price, and pool identity features
- a **calibration step** to improve the numerical reliability of predicted probabilities

The final paper uses an **EWMA-based baseline**, a **Histogram Gradient Boosting** residual model, and **temperature scaling** for calibration.

### 5. Evaluate decision usefulness, not just prediction accuracy

CAI is evaluated both as a forecasting model and as a decision tool. The policy notebooks test whether better interruption-risk estimates lead to better pool choices under two objectives:

- **Price-risk tradeoff:** choose pools that balance hourly cost and interruption risk
- **Retry-cost objective:** choose pools that minimize expected total cost when failed jobs can be retried

## Main findings from the paper

The paper reports several key results:

- Near-term spot interruption risk is **predictable enough to model meaningfully**.
- The final CAI model outperforms simpler baselines such as EWMA-only, rolling 24-hour averages, fixed interruption mappings, and global averages on the main forecasting task.
- A **baseline-plus-residual** formulation performs much better than a direct one-step machine learning alternative.
- **Histogram Gradient Boosting** was the strongest of the candidate tree-based models studied.
- The adaptive probing strategy improves downstream prediction slightly over the capped uniform strategy, though the improvement is modest.
- Under a linear price-risk objective, CAI leads to better realized decisions than the baseline strategies tested.
- Under the retry-cost objective, a simpler 24-hour average can sometimes perform better when the key challenge is ranking a very small set of candidate pools.

That last result is important: stronger overall probabilistic forecasting does not automatically guarantee the best result for every downstream policy objective.

## Repository structure

The repository is organized around the three major parts of the project.

```text
Cloud-Arbitrage-Index/
├── data/
│   ├── probe_results_combined.csv
│   ├── probe_results_rows.csv
│   └── probe_results_topup_rows.csv
├── src/
│   ├── probing/
│   │   ├── baseline_scheduler.py
│   │   ├── bandit_scheduler.py
│   │   └── cai_bandit.py
│   └── cai_model/
│       ├── CAI_Model.ipynb
│       └── evaluation/
│           ├── CAI_Learning_Curve.ipynb
│           ├── CAI_ML_Model_Comparison.ipynb
│           ├── CAI_Probing_Strategy_Comparison.ipynb
│           └── policies/
│               ├── Price_Risk_Policy.ipynb
│               └── Retry_Cost_Policy.ipynb
└── README.md
```

## Repository contents

### `src/probing/`

This directory contains the live probing and scheduling code used to collect AWS Spot data.

- **`baseline_scheduler.py`**: runs the uniform capped probing policy and handles probe execution, logging, AWS interaction, and top-up behavior.
- **`bandit_scheduler.py`**: runs the adaptive scheduler used to allocate probes based on uncertainty.
- **`cai_bandit.py`**: contains the bandit logic and posterior-based uncertainty machinery used by the adaptive policy.

The probing code interacts directly with AWS and stores results in a backend that includes Supabase logging.

### `src/cai_model/`

This directory contains the main modeling notebook and evaluation notebooks.

- **`CAI_Model.ipynb`**: the primary modeling workflow used to build the final interruption-risk model.
- **`evaluation/CAI_Learning_Curve.ipynb`**: evaluates how performance changes as more historical training data are added.
- **`evaluation/CAI_ML_Model_Comparison.ipynb`**: compares model classes such as Decision Tree, XGBoost, and Histogram Gradient Boosting.
- **`evaluation/CAI_Probing_Strategy_Comparison.ipynb`**: compares downstream performance using data collected from different probing policies.
- **`evaluation/policies/Price_Risk_Policy.ipynb`**: evaluates the linear price-risk decision objective.
- **`evaluation/policies/Retry_Cost_Policy.ipynb`**: evaluates the retry-based decision objective.

### `data/`

This directory contains the probe logs used in the notebook workflows.

- **`probe_results_rows.csv`**: baseline probing results
- **`probe_results_topup_rows.csv`**: top-up / additional probing results
- **`probe_results_combined.csv`**: merged dataset used by the main model notebook

## Data and feature notes

The paper describes the final modeling dataset as containing roughly **7,500 one-hour probe outcomes**, ordered chronologically by run start time. The feature set is intentionally restricted to information available at launch time, including:

- provider, region, and instance type
- current spot price
- normalized recent price behavior
- cyclic time-of-day and day-of-week features
- weekend indicator and coarse time block
- recent pool-level interruption statistics used in the baseline estimate

This keeps the pipeline leakage-safe and aligned with the real operational decision point.

## How to use this repository

This repository is currently notebook-centered rather than packaged as a polished installable library. The most practical way to work with it is:

1. Start with **`src/cai_model/CAI_Model.ipynb`** for the main modeling pipeline.
2. Use the notebooks in **`src/cai_model/evaluation/`** to reproduce the analyses in the paper.
3. Use the files in **`src/probing/`** only if you want to study or extend the live probe-collection system.

Because the probing scripts interact with AWS resources and an external Supabase backend, they should be reviewed carefully before being run in a new environment.

## Requirements

The repository includes both notebook-based analysis and live cloud-probing code, so the exact environment depends on what you are trying to run.

For the modeling notebooks, you will generally need:

- Python 3.x
- Jupyter / Google Colab
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

For the probing scripts, you will also need:

- AWS credentials with permission to launch and terminate EC2 Spot Instances
- `boto3`
- `supabase`
- any environment variables or credentials expected by the scripts

## Scope and limitations

This repository reflects a research implementation developed for a Princeton independent work thesis. It is designed first as an empirical and methodological contribution, not as a production-ready cloud scheduling platform.

Important limitations noted in the paper include:

- the study focuses only on AWS
- the empirical evaluation uses only six pools
- the dataset covers a limited time window under a fixed budget
- the model predicts fixed-horizon risk rather than full time-to-failure curves
- policy performance depends on the downstream decision objective
