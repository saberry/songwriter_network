import pandas as pd

df = pd.read_csv('songwriter_summary.csv')

print('Sample songwriters with their date ranges:')
print()
print('Top 5 by duration:')
for idx, row in df.nlargest(5, 'duration_years').iterrows():
    print(f'{row["songwriter"]:20s} - {row["duration_years"]:6.2f} years ({row["duration_days"]:.0f} days), {row["num_appearances"]} appearances')
print()
print('Some classic songwriters:')
for songwriter in ['I.Berlin', 'J.Rodgers', 'R.Isley', 'J.Scott']:
    matches = df[df['songwriter'] == songwriter]
    if len(matches) > 0:
        row = matches.iloc[0]
        print(f'{row["songwriter"]:20s} - {row["duration_years"]:6.2f} years ({row["duration_days"]:.0f} days), {row["num_appearances"]} appearances')
