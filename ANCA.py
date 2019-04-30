import numpy as np
import networkx as nx
import pandas as pd
import math
from sklearn.cluster import KMeans

s='sample.csv'
sn='sampleNode.CSV'

class ANCA:
    def __init__(self,edgeData,nodeData,upper,lower):
        self.edgeData=edgeData
        self.nodeData=nodeData
        self.upper=upper
        self.lower=lower


    def build_graph(self):
        '''take two data sets as input, set node attribute,
            return G'''
        df= pd.read_csv(self.edgeData, sep=',')
        G = nx.from_pandas_edgelist(df,'s','t')
        df2=pd.read_csv(self.nodeData,sep=',')
        self.vcount=df2.shape[0]
        for i,attr in enumerate(list(df2.columns)):
            nx.set_node_attributes(G,df2[attr],'attr'+str(i))
        return G

    def detect_seed(self,G):
        '''return a set of nodes that include self.upper fraction of largest centrality, and self.lower faction of
            the smallest centrality'''
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
        '''characterize each node according to its relation(shortest path) to every seed node,
            return a R^(V*S) matrix'''
        member_m=[]
        for v in self.G.nodes:
            row=[]
            for s in self.seeds:
                row.append(nx.shortest_path_length(self.G,v,s))
            member_m.append(row)
        return np.array(member_m)

    def  build_attriMaxtrix(self):
        '''characterize each node according to its attribute relation to every other node,
            return a R^(V*V) matrix'''
        attri_m=[]
        for i in self.G.nodes:
            row=[]
            for j in self.G.nodes:
                row.append(self.euler(i,j))
            attri_m.append(row)

        return np.array(attri_m)

    def euler(self,a,b):
        '''Eucledian Difference'''
        a=self.G.nodes[a]
        b=self.G.nodes[b]
        attList=list(a)
        sum=0
        for att in attList:
            sum+=(a[att]-b[att])**2
        return math.sqrt(sum)



    def anca_calc(self,k=None):
        '''main calc'''
        self.G=self.build_graph() #build Graph
        self.seeds=self.detect_seed(self.G)#build seeds set

        topM=self.build_memberMatrix()#topM = topological information
        attM=self.build_attriMaxtrix()#attM = attribute information

        l1=self.svd(topM) #svd both matrices
        l2=self.svd(attM)

        featureSpaceX=np.column_stack((l1,l2)) #stack the feature space together
        featureSpaceY=self.featureY(featureSpaceX)

        if k == None: #recommended k for kMeans cluster
            k=int(math.sqrt(self.vcount/2))

        cluster=KMeans(n_clusters=k, random_state=0).fit_predict(featureSpaceY)
        return self.cluster(cluster)

    def cluster(self,cluster):
        '''assign each node to a set'''
        ans={}
        for index,cat in enumerate(cluster):
            if cat not in ans:
                ans[cat]=set()
            ans[cat].add(index)
        return ans


    def svd(self,M,k=2):
        '''svd a given matrix, k = top k largest eigenvectors'''
        u, z, v = np.linalg.svd(M, full_matrices=False)
        z=np.diag(z)
        v=np.dot(z,v)
        l=np.dot(u[:,:k],v[:k,:])
        return l

    def featureY(self,featureSpaceX):
        '''normalize the attribute of each node'''
        featureSpaceY=[]
        for row in featureSpaceX:
            newRow=[]
            sm=sum(row)
            for i in row:
                newRow.append(i/sm)
            featureSpaceY.append(newRow)
        return featureSpaceY



a=ANCA(s,sn,0.2,0.1)
cluster=a.anca_calc()
print(cluster)