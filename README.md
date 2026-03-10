
# opt_inv_portfolios

Portfolio-optimisation experiments on S&P 500 historical data using **Simulated Annealing** and **Tabu Search**, evaluated with three Monte Carlo–based objective functions:

- **Value at Risk (VaR 95%)** (one-tailed, based on simulated portfolio gains)
- **(Adapted) Sharpe ratio** (mean / std of simulated gains)
- **(Adapted) Maximum Drawdown (MDD)** (based on min/max simulated gains)

The workflow:
1. Load price data from `SP500_data.csv`
2. Aggregate daily prices into “weekly” log-returns (5 trading days per block)
3. Compute per-asset mean and standard deviation on weekly log-returns
4. Run metaheuristics to search for an allocation that maximises the selected objective


## Project structure

- `main.py` — all code (data loading, aggregation, objective functions, SA/TS implementations, and experiment runner that exports results to Excel)
- `SP500_data.csv` — input dataset (daily prices; the script skips the header row and removes the first column)
- `Results_improvments.xlsx` — results spreadsheet (generated/updated by the experiments)
- `report.pdf` — project report / assignment write-up


## Requirements

- Python 3.x
- `numpy`
- `pandas`

Install dependencies:
```bash
pip install numpy pandas
```


## How to run

From the project root (where `SP500_data.csv` is located):

```bash
python main.py
```

What you should expect:
- The script prints dataset shapes, the initial random allocation, and baseline metric values.
- It then runs:
  - Simulated Annealing for VaR / Sharpe / MDD
  - Tabu Search for VaR / Sharpe / MDD
- In the “Session 3” section, it runs multiple configurations and writes/updates an Excel file (by default `Results_improvements_refactored.xlsx` as currently coded).

> Note: your repo contains `Results_improvments.xlsx`, but the code currently writes to `Results_improvements_refactored.xlsx` (different filename). If you want it to write to the existing file, change `output_excel_file` in `main.py`.


## Data notes

- `load_data()` reads `SP500_data.csv` with `skip_header=1` and then removes the first column (`data[:, 1:]`).
- `aggregate_data()` groups the time series into blocks of **5 trading days** and computes a log-ratio:
  \[
  \log\left(\frac{P_{end}}{P_{start}}\right)
  \]
  producing a weekly return-like series for each asset.


## Algorithms implemented

### Neighbour generation
A neighbour is created by moving **5%–10%** of the capital from one currently-invested asset to a (possibly new) destination asset.

### Simulated Annealing (SA)
- Accepts improving moves always
- Accepts worsening moves with probability `exp(-Δ/T)`
- Temperature decreases by `T *= alpha` each iteration

### Tabu Search (TS)
- Maintains a tabu list keyed by rounded solutions
- Uses an aspiration criterion to allow tabu moves if they outperform the best by a threshold
- Can do random restart if no admissible neighbours are found


## Output

- Console logs of baseline and best evaluations
- Excel sheet with rows:
  - `VaR at 95%`
  - `Sharp`
  - `MDD`
  - `Execution Time`
- Columns for each configuration (base case + improvements)


## Reproducibility

A fixed RNG seed is set in the script (`np.random.seed(0)`), so repeated runs produce the same random initial solution and Monte Carlo draws unless you change/comment that line.


## Notes / known limitations

- Objective functions rely on normality assumptions for returns (Monte Carlo sampling from `Normal(mean, std)` per asset).
- Allocations are represented in absolute “capital units” (default `amount = 100`), not portfolio weights that sum to 1.
- The VaR objective currently returns `sum(solution) + VaR_95`, so it behaves like “capital plus 5th percentile gain”; adjust if you want a conventional VaR definition/sign convention.
