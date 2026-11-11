import pandas as pd

df = pd.read_csv('songwriter_summary.csv')

# Show some insights
print('=' * 80)
print('KEY INSIGHTS FROM SONGWRITER DATA')
print('=' * 80)
print()
print(f'Total unique songwriters: {len(df):,}')
print(f'Total collaborations (sum of num_collaborators): {int(df["num_collaborators"].sum()):,}')
print(f'Total appearances (sum): {int(df["num_appearances"].sum()):,}')
print()
print('Top 10 most connected songwriters (by collaborators):')
print(df.nlargest(10, 'num_collaborators')[['songwriter', 'num_collaborators', 'num_appearances']].to_string(index=False))
print()
print('Top 10 songwriters with best average debut position (lower number = better/earlier):')
print(df.nsmallest(10, 'avg_debut_position')[['songwriter', 'avg_debut_position', 'num_appearances']].to_string(index=False))
print()
print('Songwriters with 0 collaborators:')
zero_collab = df[df['num_collaborators'] == 0]
print(f'Count: {len(zero_collab)}')
print(zero_collab.head(10)[['songwriter', 'num_collaborators', 'num_appearances']].to_string(index=False))
