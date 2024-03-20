from datetime import datetime
import numpy as np
import pandas as pd

from utilities import *
import opendp.prelude as dp

dp.enable_features("contrib", "floating-point", "honest-but-curious")

# PUBLIC INFO
start_date, end_date = datetime(2020, 9, 1), datetime(2021, 3, 31)
time_col = "date"

# DATA
path = "synth_data.csv"
df = pd.read_csv(path)

t_pre = (
    make_preprocess_sum_cols()
    >> make_preprocess_location()
    >> make_truncate_time(start_date, end_date, time_col)
)
d_pre = t_pre.map(1)

pq_losses = [(1.0, 0.0), (2.0, 5e-7), (1.0, 1e-7)]

m_global = t_pre >> dp.c.make_sequential_composition(
    input_domain=dataframe_domain(),
    input_metric=dp.symmetric_distance(),
    output_measure=dp.fixed_smoothed_max_divergence(T=float),
    d_in=d_pre,
    d_mids=pq_losses,
)

# define the global compositor, then delete the data/remove access
qbl_compositor = m_global(df)
del df

## Epsilon we will spend for this preprocessing stage
preproc_epsilon, preproc_delta = pq_losses[0]
pq_preproc_epsilon = np.nextafter(np.nextafter(preproc_epsilon / 4.0, -1), -1)

m_preproc = dp.c.make_sequential_composition(
    input_domain=dataframe_domain(),
    input_metric=dp.symmetric_distance(),
    output_measure=dp.max_divergence(T=float),
    d_in=d_pre,
    d_mids=[pq_preproc_epsilon] * 4,
)
qbl_preproc = qbl_compositor(dp.c.make_pureDP_to_fixed_approxDP(m_preproc))


## 1.1 DATA SIZE
m_count = (
    make_select_column("amt")
    >> dp.t.then_count()
    >> dp.m.then_laplace(d_pre / pq_preproc_epsilon)
)
size_guess = qbl_preproc(m_count)
print("dp data size:", size_guess)


# 1.2 TUKEY FENCES
## I imagine $ 10 mil a week isn't going through many terminals...
bounds_est_amt = 0.0, 10000000.0
## I imagine 100k transactions a week isn't going through many terminals...
bounds_est_num_transactions = 0.0, 100000.0

## Tukey Fences: 0.25 & 0.75 quantiles overall for the amt & nb_transactions
## grouping column with a desired relative error ratio: counts in each bin & merchant-level median per industry
b = dp.t.choose_branching_factor(size_guess=size_guess)
# num_bins with good results: 1000000

## I imagine $ 10 mil a week isn't going through many terminals...
bounds_est_amt = 0.0, 10000000.0
## I imagine 100k transactions a week isn't going through many terminals...
bounds_est_num_transactions = 0.0, 100000.0

m_amt_quantiles = make_tree_quantiles(
    [0.25, 0.75], b, bounds_est_amt, d_pre, pq_preproc_epsilon
)
m_nb_transactions_quantiles = make_tree_quantiles(
    [0.25, 0.75], b, bounds_est_num_transactions, d_pre, pq_preproc_epsilon
)

amt_quantiles = qbl_preproc(make_select_column("amt") >> m_amt_quantiles)
nb_transactions_quantiles = qbl_preproc(
    make_select_column("nb_transactions") >> m_nb_transactions_quantiles
)

amt_iqr = amt_quantiles[1] - amt_quantiles[0]
nb_transactions_iqr = nb_transactions_quantiles[1] - nb_transactions_quantiles[0]

# Define upper bounds for potential outliers
amt_upper_bound = amt_quantiles[1] + 1.5 * amt_iqr
nb_transactions_upper_bound = nb_transactions_quantiles[1] + 1.5 * nb_transactions_iqr

bounds_amt = (0.0, amt_upper_bound)
bounds_nb_transactions = (0.0, nb_transactions_upper_bound)
print("dp amt upper bound:", amt_upper_bound)
print("dp nb_transactions upper bound:", nb_transactions_upper_bound)

# 1.3 GROUPING COLUMNS
candidates = [
    ["date", "merch_category", "transaction_type"],
    ["date", "merchant_postal_code_1"],
    ["date", "merchant_postal_code_1", "merch_category"],
    ["date", "merchant_postal_code_1", "merch_category", "transaction_type"],
]

m_select_gcols = make_select_grouping_cols(
    candidates=candidates,
    min_bin_size=89,
    d_in=d_pre,
    d_out=pq_preproc_epsilon,
)
grouping_columns = qbl_preproc(m_select_gcols)

print("dp selected grouping columns:", grouping_columns)

# 2. GROUPING KEYS
m_private_keys = make_threshold_counts_budget(
    by=grouping_columns, d_in=d_pre, d_out=pq_losses[1], delta_p=.25e-7
)
counts = qbl_compositor(m_private_keys)

print("dp counts:", counts)

# 3. SUMS
sums_loss = pq_losses[2]


query_constructors = [
    lambda s: make_private_sum_by(
        column="amt",
        by=grouping_columns,
        bounds=bounds_amt,
        scale=s,
    ),
    lambda s: make_private_sum_by(
        column="nb_transactions",
        by=grouping_columns,
        bounds=bounds_nb_transactions,
        scale=s,
    ),
]


# how large each scale parameter should be relative to each other to make the budgets come out equal
#                         gives each query parity                  * distribution of budget
relative_scales = find_relative_scales(query_constructors, d_in=d_pre) * [1, 1]


def make_analysis(param):
    return at_delta(
        make_merge_keyset(counts.index)
        >> dp.c.make_basic_composition(
            [
                const(weight * param)
                for weight, const in zip(relative_scales, query_constructors)
            ]
        ),
        delta=sums_loss[1],
    )


scale_dp = dp.binary_search_param(make_analysis, d_in=d_pre, d_out=sums_loss, T=float)
print("sum noise scales:", scale_dp * relative_scales)

m_analysis = dp.binary_search_chain(make_analysis, d_in=d_pre, d_out=sums_loss, T=float)
assert m_analysis.map(d_pre) <= sums_loss
noisy_aggregated_amt, noisy_aggregated_nb_transactions = qbl_compositor(m_analysis)

print("dp sum nb_transactions:", noisy_aggregated_nb_transactions)
print("dp sum amt:", noisy_aggregated_amt)
