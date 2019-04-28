import random
import numpy as np
import networkx as nx
import pandas as pd
from numpy import linalg as LA
import math

s='sample.csv'
sn='sampleNode.CSV'
class ANCA:
    def __init__(self,edgeData,nodeData,k,upper,lower):
        self.edgeData=edgeData
        self.nodeData=nodeData
        self.k=k
        self.upper=upper
        self.lower=lower


    def build_graph(self):
        df= pd.read_csv(self.edgeData, sep=',')
        G = nx.from_pandas_edgelist(df,'s','t')
        df2=pd.read_csv(self.nodeData,sep=',')
        self.vcount=df2.shape[0]
        for i,attr in enumerate(list(df2.columns)):
            nx.set_node_attributes(G,df2[attr],'attr'+str(i))
        return G

    def detect_seed(self,G):
        seeds=set()
        top=int(self.vcount- self.vcount*self.upper)
        low=int(self.vcount*self.lower)
        for cenList in [nx.eigenvector_centrality(G),nx.degree_centrality(G),nx.closeness_centrality(G)]:
            sortedList=sorted(cenList.items(), key=lambda  kv:kv[1])
            front=[x[0] for x in sortedList[:low]]
            tail=[x[0] for x in sortedList[top:]]
            for i in front+tail:
                seeds.add(i)
        return seeds

    def build_memberMatrix(self):
        member_m=[]
        for v in self.G.nodes:
            row=[]
            for s in self.seeds:
                row.append(nx.shortest_path_length(self.G,v,s))
            member_m.append(row)
        return np.array(member_m)

    def  build_attriMaxtrix(self):
        attri_m=[]
        for i in self.G.nodes:
            row=[]
            for j in self.G.nodes:
                row.append(self.euler(i,j))
            attri_m.append(row)

        return np.array(attri_m)

    def euler(self,a,b):
        a=self.G.nodes[a]
        b=self.G.nodes[b]
        attList=list(a)
        sum=0
        for att in attList:
            sum+=(a[att]-b[att])**2
        return math.sqrt(sum)



    def anca_calc(self):
        self.G=self.build_graph()
        self.seeds=self.detect_seed(self.G)
        topM=self.build_memberMatrix()
        a=np.linalg.svd(topM)

        print(a)

        # vv,vc=np.linalg.eig(svd)
        # print(vv)
        # print(vc)
        # attM=self.build_attriMaxtrix()
        # atVec=np.linalg.eig(attM)
        # print(atVec[0])
        # print(atVec[1])




a=ANCA(s,sn,2,0.2,0.1)
a.anca_calc()