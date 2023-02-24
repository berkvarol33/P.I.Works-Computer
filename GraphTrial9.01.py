# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:15:43 2023

@author: berk.varol
"""
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from pandas import read_excel as rc
import math
from haversine import haversine
from scipy.sparse import coo_matrix
#HARVESINE TAKES DISTANCE DATA IN ACCORDANCE TO EARTHS SHAPE!

#reading the excel sheet
df = pd.read_excel("Dataset.xlsx", header=0)
print (df)


#taking nodes
nodes1=df.loc[:,['Site_A','Latitude_A','Longitude_A']] #.loc to export spesific rows of data
nodes2=df.loc[:,['Site_B','Latitude_B','Longitude_B']]

# changing the column headers to merge them together
nodes1.columns = ['SiteName', 'Latitude', 'Longitude']
nodes2.columns = ['SiteName', 'Latitude', 'Longitude']

#merge nodes data
nodes=pd.concat([nodes1,nodes2],axis=0)
nodes = nodes.reset_index(drop=True) # index resetliyo

#if in future we will have duplicates (one node connecting to several) this 
#will help us drop the duplicates.
nodes=nodes.drop_duplicates()
print(nodes)


# Define the graph -nodes info part
G = nx.Graph()

node_names =np.array(nodes.loc[:,'SiteName'])
lats = np.array(nodes.loc[:,'Latitude'].tolist())
lons = np.array(nodes.loc[:,'Longitude'].tolist())



#add node
for i in range(0, len(nodes)):
    node_name=node_names[i]
    lat=lats[i]
    lon=lons[i]
    G.add_node(node_name, pos=(lat,lon))

    
    
#edges info part
weights=[]

node_name_as=np.array(df.loc[:,'Site_A'])
node_name_bs =np.array(df.loc[:,'Site_B'])    

for i in range (0, len(df)):
    lat1=df.loc[i,'Latitude_A']
    lat2=df.loc[i,'Latitude_B']
    lon1=df.loc[i,'Longitude_A']
    lon2=df.loc[i,'Longitude_B']
    
    dist=haversine((lat1, lon1), (lat2, lon2))
    weights=np.append(weights, dist)    
    
    node_name_a=node_name_as[i]
    node_name_b=node_name_bs[i]
    weight=weights[i]
    G.add_edge(node_name_a,node_name_b,weight=weight)

print(G)
# Extract the node positions
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
plt.figure(3,figsize=(24,24)) 
nx.draw(G,pos, with_labels=True,node_size=60,font_size=2)


plt.show()