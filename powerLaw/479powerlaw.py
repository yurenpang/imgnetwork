import pandas as pd
import numpy as np
import networkx as nx
file='acna_knn_edges.csv'

outf='outdegree.txt'
inf='indegree.txt'

df=pd.read_csv(file)
G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr = ['weights'],create_using=nx.DiGraph())
outf = open(outf, 'w')
inf=open(inf,'w')
for v in G.nodes:
    inf.write(str(G.in_degree(v)))
    inf.write('\n')
    outf.write(str(G.out_degree(v)))
    outf.write('\n')
print('finished')



