import pandas as pd
import numpy as np
from teneto import TemporalNetwork
from teneto.networkmeasures import (
    bursty_coeff, 
    intercontacttimes,
    temporal_closeness_centrality,
    temporal_betweenness_centrality,
    reachability_latency
)
import teneto.communitydetection as cd
from teneto.communitymeasures import promiscuity
from bursty_dynamics.scores import calculate_scores
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
temp_net_df_raw = pd.read_csv("songwriter_connections_time.csv")
temp_net_df = pd.read_csv("songwriter_connections_time.csv")
temp_net_df['date'] = pd.to_datetime(temp_net_df['date'], format='%Y-%m-%d')
temp_net_df = temp_net_df[temp_net_df['date'] > pd.Timestamp('2000-01-01')]

temp_net_df['date'] = temp_net_df['date'] + pd.to_timedelta(
    np.random.randint(-10, 10, size=len(temp_net_df)), unit="D"
)

temp_net_df = temp_net_df.rename(columns={"Source": "i", "Target": "j", "date": "t", "Weight": "w"})

# Create mappings for nodes and times
i_name_number = {name: i for i, name in enumerate(set(temp_net_df["i"]).union(set(temp_net_df["j"])))}
j_name_number = {name: j for j, name in enumerate(set(temp_net_df["i"]).union(set(temp_net_df["j"])))}
t_name_number = {name: t for t, name in enumerate(sorted(set(temp_net_df["t"])))}

# Map names to numbers
temp_net_df["i"] = temp_net_df["i"].map(i_name_number)
temp_net_df["j"] = temp_net_df["j"].map(j_name_number)
temp_net_df["t"] = temp_net_df["t"].map(t_name_number)

temp_net_df = temp_net_df[["i", "j", "t", "w"]].astype('int32')

# Create TemporalNetwork
tnet = TemporalNetwork(
    from_df=temp_net_df, 
    nettype='wu',
    timelabels=list(t_name_number.keys()), 
    nodelabels=list(i_name_number.keys()),
    timetype='discrete', 
    timeunit='year'
)

df_filtered = temp_net_df[temp_net_df.groupby('i')['t'].transform('count') >= 20]
df_filtered = temp_net_df[temp_net_df.groupby('j')['t'].transform('count') >= 20]

bursty_tnet = TemporalNetwork(
    from_df=df_filtered, 
    nettype='wu',
    timelabels=list(t_name_number.keys()), 
    nodelabels=list(i_name_number.keys())
)

# Create reverse mapping for displaying results
number_to_name = {v: k for k, v in i_name_number.items()}

# Bursty coefficient (how concentrated collaborations are in time)
score_df, scatter_plot = calculate_scores(temp_net_df, subject_id='i', time_col='t', scatter=True)
# Jim Vallence is the burstiest node, with a value of .86, which is very high. 
# This means that Ja Rule's collaborations are highly concentrated in time, 
# with many collaborations happening in a short period and then long periods of inactivity.
# Ja Rule is also pretty high

# Nathan Chapman is the least with a value of -1.0, 
# indicating that his collaborations are more evenly spread out over time.

# NETWORK-LEVEL STATISTICS
network_size = tnet.N
num_time_periods = tnet.T
time_range = (list(t_name_number.keys())[0], list(t_name_number.keys())[-1])
total_edges = len(temp_net_df)
total_weight = temp_net_df['w'].sum()

# Global density
density_values = []
for t in range(tnet.T):
    possible_edges = tnet.N * (tnet.N - 1) / 2
    actual_edges = len(temp_net_df[temp_net_df['t'] == t])
    if possible_edges > 0:
        density_values.append(actual_edges / possible_edges)

average_density = np.mean(density_values)

import networkx as nx
from networkx.algorithms import community

G = nx.from_pandas_edgelist(
    temp_net_df_raw, 
    source='Source', 
    target='Target', 
    edge_attr=['Weight', 'date'] 
)

# Communities #
louvain_communities = community.louvain_communities(G, seed=42)

print(f"Detected {len(louvain_communities)} communities using Louvain.")
for idx, comm in enumerate(louvain_communities):
    print(f"Community {idx}: {sorted(list(comm))}")


# Calculate individual contact degrees (node promiscuity)
degrees = dict(G.degree())

# Check for "assortative mixing" (Do highly active nodes stick together?)
# Returns a value between -1 and 1. Negative means promiscuous nodes connect to isolated nodes.
assortativity = nx.degree_assortativity_coefficient(G)
print(f"Network Assortativity: {assortativity}")

# Promiscuity #
import community as community_louvain
partition = community_louvain.best_partition(G)

promiscuity_scores = {}
for node in G.nodes():
    node_degree = G.degree(node)
    if node_degree == 0:    
        promiscuity_scores[node] = 0
        continue
        
    # Count how many links this node has to each community
    community_link_counts = {}
    for neighbor in G.neighbors(node):
        com = partition[neighbor]
        community_link_counts[com] = community_link_counts.get(com, 0) + 1
        
    # Apply the participation coefficient formula
    sum_squares = sum((count / node_degree) ** 2 for count in community_link_counts.values())
    promiscuity_scores[node] = 1 - sum_squares

# Find the most "promiscuous" hubs
most_promiscuous = sorted(promiscuity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top promiscuous nodes:", most_promiscuous)

promiscuity_df = pd.DataFrame(list(promiscuity_scores.items()), columns=['Node', 'Promiscuity'])
sns.histplot(promiscuity_df['Promiscuity'], bins=20)
plt.show()
# Modularity #

import networkx.algorithms.community as nx_comm
# communities = [{node1, node2}, {node3, node4}]
modularity = nx_comm.modularity(G, louvain_communities)

betweeness_centrality = nx.betweenness_centrality(G)
# The higher the betweenness centrality, the more a node acts as a bridge between other nodes in the network.
# 
top_betweenness = sorted(betweeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
bottom_betweenness = sorted(betweeness_centrality.items(), key=lambda x: x[1], reverse=False)[:10]
degree_centrality = nx.degree_centrality(G)
# find top 10 degree centrality nodes
top_degree_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
clustering_coefficient = nx.clustering(G)

from networkx.algorithms.community import partition_quality
coverage, performance = partition_quality(G, louvain_communities)

import scipy.stats as stats

standardized_data = (promiscuity_df['Promiscuity'] - promiscuity_df['Promiscuity'].mean()) / promiscuity_df['Promiscuity'].std()
ks_stat, ks_p = stats.kstest(standardized_data.dropna(), 'norm')
print(f"KS Test P-value: {ks_p}")

joblib.dump(
    [louvain_communities, modularity, assortativity, 
    promiscuity_scores, betweeness_centrality, degree_centrality, 
    clustering_coefficient], 
    "network_analysis_results.pkl"
)

import joblib
louvain_communities, modularity, assortativity, promiscuity_scores, betweeness_centrality, degree_centrality, clustering_coefficient = joblib.load("network_analysis_results.pkl")



# NODE-LEVEL MEASURES
# Temporal Closeness Centrality
closeness_dict = temporal_closeness_centrality(tnet)
top_closeness = sorted(closeness_dict.items(), key=lambda x: x[1], reverse=True)[:10]

# Temporal Betweenness Centrality
betweenness_dict = temporal_betweenness_centrality(tnet)
top_betweenness = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:10]

# Degree/Strength statistics
strength_dict = {}
degree_dict = {}
for node in range(tnet.N):
    strength = 0
    degree = 0
    for t in range(tnet.T):
        edges_out = temp_net_df[(temp_net_df['i'] == node) & (temp_net_df['t'] == t)]['w'].sum()
        edges_in = temp_net_df[(temp_net_df['j'] == node) & (temp_net_df['t'] == t)]['w'].sum()
        strength += edges_out + edges_in
        if edges_out > 0 or edges_in > 0:
            degree += 1
    strength_dict[node] = strength
    degree_dict[node] = degree
top_strength = sorted(strength_dict.items(), key=lambda x: x[1], reverse=True)[:10]


# TEMPORAL DYNAMICS
# Activity over time
activity_by_time = {}
for t in range(tnet.T):
    year = list(t_name_number.keys())[t]
    activity_by_time[year] = len(temp_net_df[temp_net_df['t'] == t])

# Inter-contact times
try:
    inter_contact = intercontacttimes(tnet)
    inter_contact_stats = {
        'mean': np.mean(inter_contact),
        'median': np.median(inter_contact),
        'max': np.max(inter_contact)
    }
except:
    inter_contact_stats = None

# Node persistence (how many time periods each node is active)
node_active_periods = {}
for node in range(tnet.N):
    active_periods = set()
    for t in range(tnet.T):
        if len(temp_net_df[(temp_net_df['i'] == node) & (temp_net_df['t'] == t)]) > 0:
            active_periods.add(t)
        elif len(temp_net_df[(temp_net_df['j'] == node) & (temp_net_df['t'] == t)]) > 0:
            active_periods.add(t)
    node_active_periods[node] = len(active_periods)

persistence_stats = {
    'mean': np.mean(list(node_active_periods.values())),
    'median': np.median(list(node_active_periods.values())),
    'max': int(np.max(list(node_active_periods.values())))
}

# COMMUNITY DETECTION
tnet_community = cd.temporal_louvain(bursty_tnet, n_iter=10, njobs=10)
promiscuity_values = promiscuity(tnet_community)

# Count communities per time step
community_counts = {}
for t in range(tnet.T):
    communities_t = set(tnet_community.community[t].values())
    community_counts[t] = len(communities_t)
avg_communities = np.mean(list(community_counts.values()))

# REACHABILITY
reach = reachability_latency(tnet)
reachability_stats = {
    'mean': np.nanmean(reach),
    'median': np.nanmedian(reach)
}

# NEW vs REPEAT COLLABORATIONS
first_appearances = {}
for idx, row in temp_net_df.iterrows():
    edge = tuple(sorted([row['i'], row['j']]))
    if edge not in first_appearances:
        first_appearances[edge] = row['t']

new_collaborations_by_time = {}
repeat_collaborations_by_time = {}
for t in sorted(set(temp_net_df['t'])):
    time_edges = temp_net_df[temp_net_df['t'] == t]
    new_count = 0
    repeat_count = 0
    for idx, row in time_edges.iterrows():
        edge = tuple(sorted([row['i'], row['j']]))
        if first_appearances[edge] == t:
            new_count += 1
        else:
            repeat_count += 1
    new_collaborations_by_time[t] = new_count
    repeat_collaborations_by_time[t] = repeat_count

collaboration_stats = {
    'total_new': sum(new_collaborations_by_time.values()),
    'total_repeat': sum(repeat_collaborations_by_time.values()),
    'ratio_new': sum(new_collaborations_by_time.values()) / (sum(new_collaborations_by_time.values()) + sum(repeat_collaborations_by_time.values()))
}
