import pymizu
import pandas as pd
import networkx as nx
from pyvis.network import Network
import numpy as np
import polars as pl


hot_100 = pl.read_csv("hot_100_only.csv", infer_schema_length=100000)

hot_100 = hot_100.with_columns(
    pl.col('song').str.to_lowercase().alias('song'),
    pl.col('artist').str.to_lowercase().alias('artist'),
    pl.col('songwriter').str.to_lowercase().alias('songwriter')
)

hot_100 = hot_100.unique(subset=['song', 'songwriter'])

hot_100 = hot_100.with_columns(
    pl.col('week').str.extract(r'(\d{4}-\d{2}-\d{2})').cast(pl.Date).alias('date')
)

hot_100 = hot_100.drop_nulls(subset=['songwriter'])

hot_100['songwriter'].value_counts()

hot_100['songwriter'].unique()

hot_100 = hot_100.sort('date')

unique_dates = hot_100['date'].unique()
i = unique_dates[0]

# results = []

def get_songwriter_connections(i):
    writers = hot_100.filter(hot_100['date'] == i)[['songwriter', 'date']]
    writers = (
        writers.with_columns([
                pl.col('songwriter').str.split(", ").alias('songwriter'),
                pl.col('songwriter').str.strip_chars().alias('original_songwriter'),
            ])
            .explode('songwriter')
            .to_dummies('songwriter')
            .select(
                    pl.all().name.replace(r"songwriter_", "")
                )
    )
    writers = writers.unpivot(on=None, index=['date', 'original_songwriter'])
    writers = writers.filter(pl.col('value') > 0)
    writers = writers.pivot(
        on='variable', index=['date', 'original_songwriter'], 
        values='value', aggregate_function='first'
        )
    writers = writers.with_columns(
        pl.col(pl.UInt8).fill_null(0)
        )
    writers = writers.drop('date', 'original_songwriter')
    writers_df = writers.to_pandas()
    songT = writers_df.T
    adj_matrix = songT.dot(songT.T)
    df_edges = adj_matrix.stack().reset_index()
    df_edges.columns = ['Source', 'Target', 'Weight']
    df_edges = df_edges[(df_edges['Weight'] > 0) & (df_edges['Source'] != df_edges['Target'])]
    df_edges['date'] = i
    df_edges.reset_index(drop=True, inplace=True)
    return df_edges


with pymizu.Pool.create(10) as pool:
    m = pool.map(get_songwriter_connections, unique_dates)
    results = m.collect()

complete_results = pd.concat(results)
complete_results.to_csv("songwriter_connections_time.csv", index=False)

# long_results = pd.concat(results)

# connections = long_results.pivot(
#     on='variable', index=['date', 'time_count', 'original_songwriter'], 
#     values='value', aggregate_function='first'
# )

# connections = connections.with_columns(
#     pl.col(pl.UInt8).fill_null(0)
# )

# dates = connections['date'].unique().sort()
# dates = pl.DataFrame({'date': dates})
# dates = dates.with_row_index('t')

# connections = connections.join(dates, on='date', how='left')

# connections = connections.with_columns(
#     pl.col('t_right').alias('t')
# )

# connections = connections.drop('date', 'time_count', 'original_songwriter')

# connections.write_csv("songwriter_connections_time.csv")

# connections_df = connections.to_pandas()
# connections_df.set_index('t', inplace=True)

# connections_df = connections_df.drop(columns=['t_right'])

# songT = connections_df.T
# adj_matrix = songT.dot(songT.T)
# df_edges = adj_matrix.stack().reset_index()
# df_edges.columns = ['Source', 'Target', 'Weight']
# df_edges = df_edges[(df_edges['Weight'] > 0) & (df_edges['Source'] != df_edges['Target'])]
# df_edges.to_csv("songwriter_edges_time.csv")


# songwriters = (
#     hot_100
#     .select('songwriter', 'date')
#     .sort('date', descending=False)
#     .drop_nulls()
#     .with_row_index("row_id")
#     .with_columns([
#         pl.col('songwriter').str.split(", ").alias('songwriter')
#     ])
#     .explode('songwriter')
#     .to_dummies('songwriter')
#     #.group_by("row_id")
#     #.agg(pl.all().sum())
#     .select(
#         pl.all().name.replace(r"songwriter_", "")
#     )
#     .drop('row_id')
# )

# not_date_columns = [col for col in songwriters.columns if col != 'date']

# # drop all columns with a column sum of 1 or less
# songwriters_counts = songwriters.select(
#     pl.all().exclude('date').sum()
# )

# filtered_cols = [
#     col_name for col_name in songwriters_counts.columns 
#     if songwriters_counts[col_name].max() > 1
# ]

# songwriters_filtered = songwriters.select(
#     ['date'] + filtered_cols
# )