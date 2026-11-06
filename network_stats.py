import pandas as pd
from teneto import TemporalNetwork
from teneto.networkmeasures import bursty_coeff
import teneto.communitydetection 
from teneto.communitymeasures import promiscuity

temp_net = pd.read_csv("songwriter_connections_time.csv")

<<<<<<< HEAD
temp_net['date'] = pd.to_datetime(temp_net['date'], format='%Y-%m-%d')

temp_net = temp_net[temp_net['date'] > "2024-01-01"]
=======
temp_net['date'] = pd.to_datetime(temp_net['date'], format='%Y-%m-%d').dt.year

#temp_net = temp_net[temp_net['date'] > "2024-01-01"]
>>>>>>> 8e6ea85 (updated code, models, and results)

temp_net = temp_net.rename(columns={"Source": "i", "Target": "j", "date": "t", "Weight": "w"})

i_name_number = {name: i for i, name in enumerate(set(temp_net["i"]).union(set(temp_net["j"])))}
j_name_number = {name: j for j, name in enumerate(set(temp_net["i"]).union(set(temp_net["j"])))}
t_name_number = {name: t for t, name in enumerate(set(temp_net["t"]))}

temp_net["i"] = temp_net["i"].map(i_name_number)
temp_net["j"] = temp_net["j"].map(j_name_number)
temp_net["t"] = temp_net["t"].map(t_name_number)

temp_net = temp_net[["i", "j", "t", "w"]]

temp_net = temp_net.apply(lambda x: x.astype('int32'), axis=1)

tnet = TemporalNetwork(
    from_df=temp_net, 
    nettype='wu',
    timelabels=list(t_name_number.keys()), 
<<<<<<< HEAD
    nodelabels=list(i_name_number.keys())
    )

bursty_coeff(tnet, calc='communities')

tnet_community = teneto.communitydetection.temporal_louvain(tnet)
promiscuity(tnet_community)
=======
    nodelabels=list(i_name_number.keys()),
    timetype='discrete', 
    timeunit='year'
    )

tnet_community = teneto.communitydetection.temporal_louvain(tnet)

promiscuity(tnet_community)

bursty_coeff(tnet, calc='global')
>>>>>>> 8e6ea85 (updated code, models, and results)
