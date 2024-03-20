import opendp.prelude as dp
import numpy as np
import pandas as pd
from itertools import product

# SUPPORTING ELEMENTS


def dataframe_domain(public_key_sets=None):
    """Creates a domain representing the set of all data frames.
    
    Assumes column names and types are public information.
    Key sets optionally named for columns in `public_key_sets` are considered public information.

    Two data frames differing in their public information 
    are considered to have a data set distance of infinity.
    """
    return dp.user_domain(
        "DataFrameDomain", lambda x: isinstance(x, pd.DataFrame), public_key_sets
    )


def series_domain():
    """Creates a domain representing the set of all series.

    Assumes series name and type are public information.

    Two series differing in their public information 
    are considered to have a data set distance of infinity.
    """
    return dp.user_domain("SeriesDomain", lambda x: isinstance(x, pd.Series))


def identifier_distance():
    """Symmetric distance between the id sets."""
    return dp.user_distance("IdentifierDistance")


def approx_concentrated_divergence():
    """symmetric distance between the id sets"""
    return dp.user_distance("ApproxConcentratedDivergence()")


# UTILITIES
def at_delta(meas, delta):
    # convert from ρ to ε(δ)
    meas = dp.c.make_zCDP_to_approxDP(meas)
    # convert from ε(δ) to (ε, δ)
    return dp.c.make_fix_delta(meas, delta)


def find_relative_scales(constructors, d_in):
    return np.array(
        [dp.binary_search_param(const, d_in, 1.0, T=float) for const in constructors]
    )


# CONSTRUCTORS


def make_preprocess_sum_cols():
    """Create a 1-stable data frame transformation that preprocesses `nb_transactions` and `amt`"""
    def function(df):
        df = df.copy()
        df["nb_transactions"] = df["amt"] / df["avg_amt"]
        df.drop("avg_amt", axis=1, inplace=True)
        df["amt"] = df["amt"].astype("float64")
        df["nb_transactions"] = df["nb_transactions"].astype("float64")
        return df

    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=identifier_distance(),
        output_domain=dataframe_domain(),
        output_metric=identifier_distance(),
        function=function,
        stability_map=lambda d_in: d_in,
    )


def make_preprocess_location():
    """Create a 1-stable transformation to bin `merch_postal_code` by city"""

    def categorize_city(code):
        if code.startswith("5"):
            return "Medellin"
        elif code.startswith("11"):
            return "Bogota"
        elif code.startswith("70"):
            return "Brasilia"
        else:
            return "Santiago"

    def preprocess_postal_code(df, unique_postal_code_1):
        new_df = df.copy()
        for city in unique_postal_code_1:
            if city == "Medellin":
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_2"
                ] = new_df["merch_postal_code"].str[-2]
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_3"
                ] = new_df["merch_postal_code"].str[-1]
            elif city == "Bogota":
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_2"
                ] = new_df["merch_postal_code"].str[2:4]
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_3"
                ] = new_df["merch_postal_code"].str[4:]
            elif city == "Brasilia":
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_2"
                ] = new_df["merch_postal_code"].str[2]
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_3"
                ] = new_df["merch_postal_code"].str[3:]
            else:  # Santiago
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_2"
                ] = new_df["merch_postal_code"].str[:2]
                new_df.loc[
                    new_df["merchant_postal_code_1"] == city, "merchant_postal_code_3"
                ] = new_df["merch_postal_code"].str[2:]
        return new_df

    def location_preprocess(df):
        loc_df = df.copy()
        # Convert merchant_postal_code into str type
        loc_df["merch_postal_code"] = loc_df["merch_postal_code"].astype(str)
        # Apply the function to create a new column
        loc_df["merchant_postal_code_1"] = loc_df["merch_postal_code"].apply(
            categorize_city
        )

        unique_postal_code_1 = loc_df["merchant_postal_code_1"].unique()
        granular_df = preprocess_postal_code(loc_df, unique_postal_code_1)

        granular_df.drop("merch_postal_code", axis=1, inplace=True)

        return granular_df

    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=identifier_distance(),
        output_domain=dataframe_domain(),
        output_metric=identifier_distance(),
        function=location_preprocess,
        stability_map=lambda d_in: d_in,
    )


def make_truncate_time(start_date, end_date, time_col):
    """Create a transformation that filters the data to a given time frame.
    
    WARNING: Assumes that the data has at most one contribution per individual per week.
    """
    number_of_timesteps = (end_date - start_date).days // 7

    def time_preprocess(df):
        df = df.copy()

        # Convert time_col into datetime type
        df[time_col] = pd.to_datetime(df[time_col])

        # Filter the DataFrame based on the specified dates
        return df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=identifier_distance(),
        output_domain=dataframe_domain(),
        output_metric=dp.symmetric_distance(),
        function=time_preprocess,
        stability_map=lambda d_in: d_in * number_of_timesteps,
    )


def make_select_column(col_name, T=float):
    """Create a transformation that extracts a single column into a float numpy array"""
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=T)),
        output_metric=dp.symmetric_distance(),
        function=lambda df: df[col_name].to_numpy(),
        stability_map=lambda d_in: d_in,
    )


def make_tree_quantiles(alphas, b, bounds, d_in, d_out, num_bins=500):
    """Create a measurement that computes multi-quantiles by post-processing a CDF estimate"""
    edges = np.linspace(*bounds, num=num_bins + 1)
    bin_names = [str(i) for i in range(num_bins)]
    vec_space = dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance()

    def make_from_scale(scale):
        return (
            vec_space
            >> dp.t.then_find_bin(edges=edges)
            >> dp.t.then_index(bin_names, "0")  # bin the data
            >> dp.t.then_count_by_categories(categories=bin_names, null_category=False)
            >> dp.t.then_b_ary_tree(leaf_count=len(bin_names), branching_factor=b)
            >> dp.m.then_laplace(scale)
            >> dp.t.make_consistent_b_ary_tree(branching_factor=b)
            >> dp.t.make_quantiles_from_counts(edges, alphas=alphas)
        )

    return dp.binary_search_chain(make_from_scale, d_in, d_out)


def make_grouping_cols_score(candidates, min_bin_contributions):
    r"""Create a transformation that assesses the utility of each candidate in `candidates`.

    Try to select a set of columns to group by that will maximize the number of columns selected,
    but won't result in bins that are too sparse.

    A rough heuristic could be to score each candidate grouping set
    by the number of bins with at least `min_bin_contributions`.
    """
    candidates = [tuple(c) for c in candidates]

    def score(x: pd.DataFrame, c):
        return (x.groupby(list(c)).size() >= min_bin_contributions).sum().astype(float)

    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.linf_distance(T=float),
        function=lambda x: [score(x, c) for c in candidates],
        stability_map=float,
    )


def make_select_grouping_cols(candidates, min_bin_size, d_in, d_out):
    """Create a measurement that selects a set of grouping columns from `candidates`."""
    def make(s):
        return (
            make_grouping_cols_score(candidates, min_bin_size)
            >> dp.m.then_report_noisy_max_gumbel(s, optimize="max")
            >> (lambda idx: candidates[idx])
        )

    return dp.binary_search_chain(make, d_in, d_out, T=float)


def make_threshold_counts(by, scale, threshold, delta_p):
    """Create a measurement that privately releases the key set and partition sizes across a set of grouping columns.

    Bound comes from just before 3.2 in https://arxiv.org/pdf/2306.07884.pdf
    """
    space = dp.vector_domain(dp.atom_domain(T=float)), dp.l2_distance(T=float)
    m_gauss = dp.c.make_zCDP_to_approxDP(dp.m.make_gaussian(*space, scale))

    def function(df: pd.DataFrame):
        # print(threshold)
        exact = df.groupby(by).size()
        noisy = m_gauss(exact.to_numpy().astype(float))
        series = pd.Series(noisy, index=exact.index)
        return series[series >= threshold]

    def privacy_map(d_in):
        epsilon = m_gauss.map(np.sqrt(d_in)).epsilon(delta_p)

        # assuming each individual contributes at most one record per-bin
        distance_to_instability = threshold / scale
        delta = d_in * np.exp(-distance_to_instability**2 / 2)

        # potential alternative tighter derivation?
        # from scipy.special import erf
        # delta = d_in * (1 - erf(distance_to_instability / np.sqrt(2))) / 2

        return epsilon, delta + delta_p

    return dp.m.make_user_measurement(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        # it would be better to do this under approx zero-concentrated DP,
        # but OpenDP doesn't have δ-zCDP composition
        output_measure=dp.fixed_smoothed_max_divergence(T=float),
        function=function,
        privacy_map=privacy_map,
    )


def make_threshold_counts_budget(
    by, d_in, d_out, delta_p
) -> dp.Measurement:
    """Create a measurement that privately releases the key set and partition sizes across a set of grouping columns.
    
    This mechanism calibrates the scale and threshold 
    to meet a pre-supplied data set distance `d_in` and privacy loss parameters `d_out` in terms of (ε, δ).
    """

    def make(s, t=1e8):
        return make_threshold_counts(
            by=by, scale=s, threshold=t, delta_p=delta_p
        )

    s = dp.binary_search_param(lambda s: make(s=s), d_in, d_out)
    t = dp.binary_search_param(lambda t: make(s=s, t=t), d_in, d_out)

    return make(s=s, t=t)


def make_merge_keyset(index: pd.Index):
    """Create a 1-stable measurement that filters keys in a data frame to those in `index`.
    
    The resulting data frame can be considered to have a public key set.
    """
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dataframe_domain(public_key_sets=[index.names]),
        output_metric=dp.symmetric_distance(),
        function=lambda df: df.merge(index.to_frame(index=False)),
        stability_map=lambda d_in: d_in,
    )


def make_sum_by(column, by, bounds):
    """Create a transformation that computes the grouped bounded sum of `column`"""
    L, U = bounds

    def function(df):
        df = df.copy()
        df[column] = df[column].clip(*bounds)
        return df.groupby(by)[column].sum()

    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=series_domain(),
        output_metric=dp.l2_distance(T=float),
        function=function,
        stability_map=lambda d_in: np.sqrt(d_in) * max(abs(L), U),
    )


def make_private_sum_by(column, by, bounds, scale):
    """Create a measurement that computes the grouped bounded sum of `column`"""
    space = dp.vector_domain(dp.atom_domain(T=float)), dp.l2_distance(T=float)
    m_gauss = space >> dp.m.then_gaussian(scale)
    t_sum = make_sum_by(column, by, bounds)

    def function(df):
        exact = t_sum(df)
        noisy_sum = pd.Series(
            np.maximum(m_gauss(exact.to_numpy()), 0), index=exact.index
        )
        return noisy_sum.to_frame()

    return dp.m.make_user_measurement(
        input_domain=dataframe_domain(public_key_sets=[by]),
        input_metric=dp.symmetric_distance(),
        output_measure=dp.zero_concentrated_divergence(T=float),
        function=function,
        privacy_map=lambda d_in: m_gauss.map(t_sum.map(d_in)),
    )
