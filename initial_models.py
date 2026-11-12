import pandas as pd
import statsmodels.formula.api as smf

# summary_stats = pd.read_csv("songwriter_summary.csv")
# promiscuity_stats = pd.read_csv("artist_promiscuity.csv")

# summary_stats['songwriter'] = summary_stats['songwriter'].str.lower()
# summary_stats = summary_stats.merge(promiscuity_stats, left_on='songwriter', right_on='Node', how='left')
# summary_stats.to_csv("songwriter_summary.csv", index=False)

summary_stats = pd.read_csv("songwriter_summary.csv")
summary_stats = summary_stats.dropna()
summary_stats = summary_stats.reset_index(drop=True)
# center and scale collaborators and promiscuity for the regression model
summary_stats['Promiscuity_scaled'] = summary_stats['Promiscuity'] - summary_stats['Promiscuity'].mean()
summary_stats['collab_scaled'] = summary_stats['num_collaborators'] - summary_stats['num_collaborators'].mean()
model = smf.negativebinomial(formula="num_appearances ~ Promiscuity", data=summary_stats, loglike_method='nb2').fit(cov_type='HC3',maxiter=5000)
print(model.summary())

model = smf.negativebinomial(formula="duration_years ~ Promiscuity", data=summary_stats, loglike_method='nb2').fit(cov_type='HC3',maxiter=5000)
print(model.summary())

import networkx as nx
causal_graph = nx.DiGraph([('Promiscuity', 'num_collaborators'), ('num_collaborators', 'duration_years')])

from dowhy import gcm
from dowhy import CausalModel

model_string = '''
digraph  {
    Promiscuity -> num_collaborators;
    Promiscuity -> duration_years;
    num_collaborators -> num_appearances;
    avg_peak_position -> num_appearances;
    avg_debut_position -> num_appearances;
    num_appearances -> duration_years;
}
'''


model= CausalModel(
        data=summary_stats,
        graph=model_string, 
        treatment='Promiscuity',
        outcome='duration_years'
        )
model.view_model(layout="dot")

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression")
print("Causal Estimate is " + str(estimate.value))

import dowhy.plotter
dowhy.plotter.plot_causal_effect(estimate, summary_stats["Promiscuity"], summary_stats["duration_years"])

res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
print(res_random)

res_placebo=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)

res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)




pearson_chi2 = (model.resid_pearson ** 2).sum()
df_resid = model.df_resid
dispersion_ratio = pearson_chi2 / df_resid

import matplotlib.pyplot as plt
import numpy as np

y_pred = model.predict()

# 3. Sort values for a clean line plot (if x is unordered)
x_sorted = np.argsort(summary_stats['Promiscuity'])
x_sorted = summary_stats['Promiscuity'][x_sorted]
y_sorted = np.argsort(summary_stats['num_appearances'])
y_sorted = summary_stats['num_appearances'][y_sorted]
y_pred_sorted = y_pred[x_sorted.index]

# 4. Plot observed vs predicted
plt.scatter(summary_stats['Promiscuity'], summary_stats['num_appearances'], color='orange', label='Observed Counts', alpha=0.6)
plt.plot(x_sorted, y_pred_sorted, color='red', linewidth=2, label='Negative Binomial Fit')
plt.xlabel('X (Predictor)')
plt.ylabel('Count (Y)')
plt.title('Negative Binomial Regression Fit')
plt.legend()
plt.show()

from pyvis.network import Network
import networkx as nx


net_data = pd.read_csv("songwriter_weighted_network_edges.csv")
net_data = net_data.sample(1000, random_state=42)  # Sample 1000 edges for visualization
G = nx.from_pandas_edgelist(net_data, source='Source', target='Target', edge_attr='Weight')
net = Network(height="100vh", width="100%", bgcolor="#f0f1eb", font_color="#383d3d")
net.from_nx(G)

# Save to the same folder (or a subfolder) as your reveal.js presentation
net.write_html("small_network_graph.html")