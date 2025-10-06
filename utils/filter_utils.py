
def filter_few_datapoints(df, group_col="cow_id", min_points=4):
    """
    Remove all groups (e.g., cows) that have fewer than `min_points` datapoints.

    Args:
        df (pd.DataFrame): Input dataframe containing the grouping column.
        group_col (str): Column name identifying the group (e.g. 'cow_id').
        min_points (int): Minimum number of rows required to keep the group.

    Returns:
        pd.DataFrame: Filtered dataframe containing only valid groups.
    """
    counts = df[group_col].value_counts()
    keepers = counts[counts >= min_points].index
    filtered = df[df[group_col].isin(keepers)].copy()
    return filtered
