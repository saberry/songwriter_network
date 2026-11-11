import polars as pl


hot_100 = pl.read_csv("hot_100_only.csv", infer_schema_length=100000)

songwriters = (
    hot_100
    .unique(subset=["song", "artist"])
    .select("song", "artist", "songwriter")
    .drop_nulls(subset=["songwriter"])
    .with_columns(pl.col("songwriter").str.split(",").alias("songwriter"))
    .explode("songwriter")
    .with_columns(pl.col("songwriter").str.strip_chars())
    .filter(pl.col("songwriter") != "")
    .unique(subset=["song", "artist", "songwriter"])
)

left_writers = songwriters.select(
    "song",
    "artist",
    pl.col("songwriter").alias("Source"),
)
right_writers = songwriters.select(
    "song",
    "artist",
    pl.col("songwriter").alias("Target"),
)

df_edges = (
    left_writers
    .join(right_writers, on=["song", "artist"])
    .filter(pl.col("Source") < pl.col("Target"))
    .group_by("Source", "Target")
    .len(name="Weight")
    .sort(["Source", "Target"])
)

df_edges.write_csv("songwriter_weighted_network_edges.csv")
