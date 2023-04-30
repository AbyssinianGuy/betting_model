import pandas as pd

# Read the CSV file
odds_df = pd.read_csv('./src/data/nba_data.csv')

# Display the DataFrame
print(odds_df)

# Manipulate the DataFrame as needed
# For example, filter rows based on specific conditions
filtered_odds_df = odds_df[(odds_df['market'] == 'h2h')]

# Save the filtered data to a new CSV file
filtered_odds_df.to_csv('filtered_odds_data.csv', index=False)

# Display the filtered DataFrame
print(filtered_odds_df)
