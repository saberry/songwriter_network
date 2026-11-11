import polars as pl
import pandas as pd
import re
from itertools import combinations

# Read the data
hot_100 = pl.read_csv('hot_100_only.csv', ignore_errors=True, infer_schema_length=50000)

# Convert to pandas for easier manipulation
df = hot_100.to_pandas()

# Extract numeric values from debut and peak columns (they have HTML artifacts)
def extract_number(value):
    if pd.isna(value):
        return None
    match = re.search(r'\d+', str(value))
    return int(match.group()) if match else None

df['debut'] = df['debut'].apply(extract_number)
df['peak'] = df['peak'].apply(extract_number)

# Get unique songs per artist-song combination
df = df.drop_duplicates(subset=['song', 'artist'])

# Explode songwriters (they're comma-separated)
df_songwriters = df[['song', 'artist', 'songwriter', 'debut', 'peak']].copy()
df_songwriters = df_songwriters.dropna(subset=['songwriter'])
df_songwriters['songwriter'] = df_songwriters['songwriter'].str.split(', ')
df_songwriters = df_songwriters.explode('songwriter')

# Strip whitespace
df_songwriters['songwriter'] = df_songwriters['songwriter'].str.strip()

# Build a song -> songwriters mapping for faster lookup
song_to_songwriters = df_songwriters.groupby(['song', 'artist'])['songwriter'].apply(list).to_dict()

# Create summary statistics using groupby
appearances = df_songwriters.groupby('songwriter').size().reset_index(name='num_appearances')
avg_positions = df_songwriters.groupby('songwriter')[['debut', 'peak']].mean().reset_index()
avg_positions.columns = ['songwriter', 'avg_debut_position', 'avg_peak_position']

# Calculate collaborators more efficiently
collaborators_dict = {}
for songwriter in df_songwriters['songwriter'].unique():
    songwriter_songs = df_songwriters[df_songwriters['songwriter'] == songwriter][['song', 'artist']].values.tolist()
    collaborators = set()
    for song, artist in songwriter_songs:
        song_key = (song, artist)
        if song_key in song_to_songwriters:
            for cowriter in song_to_songwriters[song_key]:
                if cowriter != songwriter:
                    collaborators.add(cowriter)
    collaborators_dict[songwriter] = len(collaborators)

# Create collaborators dataframe
collaborators_df = pd.DataFrame(list(collaborators_dict.items()), columns=['songwriter', 'num_collaborators'])

# Merge all dataframes
summary_df = appearances.merge(collaborators_df, on='songwriter')
summary_df = summary_df.merge(avg_positions, on='songwriter')

# Sort by number of appearances descending
summary_df = summary_df.sort_values('num_appearances', ascending=False).reset_index(drop=True)

# Display info
print("Songwriter Summary Statistics")
print("=" * 80)
print(f"Total unique songwriters: {len(summary_df)}")
print()
print(summary_df.head(20))
print()

# Save to CSV
summary_df.to_csv('songwriter_summary.csv', index=False)
print("Saved to songwriter_summary.csv")

# Display some basic stats
print()
print("Summary Statistics:")
print(summary_df[['num_collaborators', 'num_appearances', 'avg_debut_position', 'avg_peak_position']].describe())
