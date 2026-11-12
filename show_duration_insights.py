import pandas as pd

df = pd.read_csv('songwriter_summary.csv')

# Show insights about duration
print('=' * 100)
print('SONGWRITER DURATION ANALYSIS')
print('=' * 100)
print()
print(f'Total unique songwriters: {len(df):,}')
print(f'Songwriters with songs spanning multiple dates: {(df["duration_days"] > 0).sum():,}')
print(f'Songwriters with only one date (duration = 0): {(df["duration_days"] == 0).sum():,}')
print()
print('Top 15 songwriters by duration (longest careers):')
print(df.nlargest(15, 'duration_years')[['songwriter', 'duration_years', 'duration_days', 'num_appearances']].to_string(index=False))
print()
print('Top 15 songwriters with multiple appearances but longest career span:')
multi_year = df[df['num_appearances'] > 1]
print(multi_year.nlargest(15, 'duration_years')[['songwriter', 'duration_years', 'num_appearances']].to_string(index=False))
print()
print('Duration Statistics:')
print(f'  Mean duration: {df["duration_years"].mean():.2f} years')
print(f'  Median duration: {df["duration_years"].median():.2f} years')
print(f'  Max duration: {df["duration_years"].max():.2f} years')
print(f'  Min duration: {df["duration_years"].min():.2f} years')
print()
print('Duration by number of appearances:')
for appearances in [1, 2, 3, 5, 10, 25, 50, 100]:
    subset = df[df['num_appearances'] >= appearances]
    if len(subset) > 0:
        print(f'  {appearances}+ appearances: {subset["duration_years"].mean():.2f} years avg ({len(subset)} songwriters)')
