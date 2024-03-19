import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder

# Define the file path # Read the dataset into a Pandas DataFrame
data = pd.read_csv('/Users/lizaskladannaya/Downloads/groceries.csv')

# Check the first few rows to ensure data is loaded correctly
#print(data.head())

# Data Cleaning: No missing values based on the provided sample, but you can check for missing values if needed
print(data.isnull().sum())

# Data Transformation: Convert the data into a transaction format
transactions = data.groupby('Member_number')['itemDescription'].apply(list).values.tolist()

# Check the first few transactions
#print(transactions[:5])

# Assuming 'transactions' is your list of transactions
# If you haven't already loaded your data, please load it first

# Count the frequency of each item
all_items = [item for transaction in transactions for item in transaction]
item_counts = Counter(all_items)

# Convert the item counts to a DataFrame for easier manipulation and plotting
item_df = pd.DataFrame.from_dict(item_counts, orient='index').reset_index()
item_df.columns = ['Item', 'Count']
item_df = item_df.sort_values(by='Count', ascending=False)

# Count the number of unique items
num_unique_items = len(set(all_items))

print("Total number of unique items:", num_unique_items)

# Plot the top N most common items
top_n = 20
plt.figure(figsize=(10, 6))
plt.barh(item_df['Item'][:top_n], item_df['Count'][:top_n])
plt.xlabel('Count')
plt.ylabel('Item')
plt.title('Top {} Most Common Items'.format(top_n))
plt.gca().invert_yaxis()  # Invert y-axis to have the highest count at the top
plt.show()


# Convert the transactions list into a one-hot encoded DataFrame
onehot_df = pd.DataFrame(columns=item_counts.keys(), data=[[item in transaction for item in item_counts.keys()] for transaction in transactions])

# Apply the Apriori algorithm to find frequent itemsets
min_support = 0.01  # Adjust the minimum support as needed
frequent_itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True)

# Generate association rules from the frequent itemsets
min_threshold = 0.5  # Adjust the minimum threshold for generating rules as needed
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_threshold)

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)  # Set to None to disable truncation of column contents

# Print the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

# Filter rules based on metrics
filtered_rules = rules[(rules['confidence'] > 0.5) & (rules['lift'] > 1.2)]

# Print the filtered rules
#print("Filtered Association Rules:")
#print(filtered_rules)
# Sort association rules by confidence in descending order
sorted_rules = filtered_rules.sort_values(by='confidence', ascending=False)

# Print the sorted rules
print("Filtered sorted Association Rules by Confidence (Descending):")
print(sorted_rules)


# Convert frozensets to strings for plotting
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

# Plot the top N most common itemsets
plt.figure(figsize=(10, 6))
top_n = 10  # Number of top frequent itemsets to display
plt.barh(frequent_itemsets['itemsets_str'][:top_n], frequent_itemsets['support'][:top_n])
plt.xlabel('Support')
plt.ylabel('Itemset')
plt.title('Top {} Frequent Itemsets'.format(top_n))
plt.gca().invert_yaxis()
plt.show()

# Filter the top N rules based on a metric (e.g., confidence, lift)
top_n_rules = sorted_rules.head(10)  # Change the number to limit the rules displayed

top_n_rules['antecedents_str'] = top_n_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
top_n_rules['consequents_str'] = top_n_rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Create a directed graph
# Create a directed graph
G = nx.DiGraph()

# Add edges for the top N association rules
for idx, rule in top_n_rules.iterrows():
    G.add_edge(rule['antecedents'], rule['consequents'], weight=rule['lift'])

# Plot the network graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # Adjust layout for better visualization
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=8, font_weight='bold', edge_color='gray', width=2, arrows=True)
plt.title('Top 5 Association Rules Network Graph')
plt.show()

# Select the top 20 most frequent itemsets
top_20_itemsets = frequent_itemsets.nlargest(20, 'support')

# Create a DataFrame for the heatmap with only the top 20 itemsets
heatmap_data = pd.DataFrame(index=top_20_itemsets['itemsets_str'], columns=item_counts.keys())

# Populate the DataFrame with support values for the top 20 itemsets
for idx, row in top_20_itemsets.iterrows():
    for item in row['itemsets']:
        heatmap_data.loc[row['itemsets_str'], item] = row['support']

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data.astype(float), cmap="YlGnBu", annot=True, fmt=".2f")
plt.title('Heatmap of Top 20 Frequent Itemsets')
plt.xlabel('Items')
plt.ylabel('Itemsets')
plt.show()

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.title('Scatter Plot of Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True)
plt.show()
