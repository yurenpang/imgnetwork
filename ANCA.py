import numpy as np
import networkx as nx
import pandas as pd
import math
from sklearn.cluster import KMeans

s='edges_with_id.csv'
sn='attributes.csv'

class ANCA:
    def __init__(self,edgeData,nodeData,upper,lower):
        self.pairDic = {}
        self.edgeData = edgeData
        self.nodeData = nodeData
        self.upper = upper
        self.lower = lower


    def build_graph(self):
        '''take two data sets as input, set node attribute,
            return G'''
        df = pd.read_csv(self.edgeData, sep=',')
        G = nx.from_pandas_edgelist(df, source='sourceID', target='targetID', edge_attr = ['weights'])

        print("###############")
        print(G.nodes())

        df2 = pd.read_csv(self.nodeData, sep=',')
        #

        self.vcount=df2.shape[0]
        self.realName_dic=df2['Country']

        for i, attr in enumerate(list(df2.columns[2:])):
            nx.set_node_attributes(G, df2[attr], 'attr'+str(i))
            print(attr)

        # G=nx.relabel_nodes(G,self.realName_dic)
        # print(G.nodes(data=True))
        # print(G.edges(data=True))

        return G

    def get_realName(self):
        return self.realName_dic

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
        """
        characterize each node according to its relation(shortest path) to every seed node,
            return a R^(V*S) matrix
        :return:
        """
        member_m = []
        for v in self.G.nodes:
            row = []
            for s in self.seeds:
                row.append(nx.dijkstra_path_length(self.G, v, s, 'w'))
            member_m.append(row)
        return np.array(member_m)

    def build_attriMaxtrix(self):
        """
        characterize each node according to its attribute relation to every other node,
        return a R^(V*V) matrix'
        :return:
        """
        attri_m = []

        for i in self.G.nodes:
            row = []
            for j in self.G.nodes:
                row.append(self.euler(i, j))
            attri_m.append(row)

        return np.array(attri_m)

    def euler(self, a, b):
        '''Eucledian Difference'''
        pair = tuple(sorted([a, b]))
        if pair in self.pairDic:
            return self.pairDic[pair]
        a=self.G.nodes[a]
        b=self.G.nodes[b]
        attList=list(a)
        sum=0
        for att in attList:
            sum+=(a[att]-b[att])**2
        self.pairDic[pair]=math.sqrt(sum)
        return math.sqrt(sum)

    def hamming(self, a, b):
        pair = tuple(sorted([a, b]))
        if pair in self.pairDic:
            return self.pairDic[pair]
        value_a = self.G.nodes[a]
        value_b = self.G.nodes[b]

        diff = 0
        for att1, att2 in zip(value_a, value_b):
            diff += abs(value_a[att1]-value_a[att2])
        self.pairDic[pair] = diff
        return diff

    def anca_calc(self,k=None):
        """

        :param k:
        :return:
        """
        self.G = self.build_graph()  # build Graph

        self.seeds = self.detect_seed(self.G)  # build seeds set

        topM = self.build_memberMatrix()  # topM = topological information
        attM = self.build_attriMaxtrix()  # attM = attribute information

        l1 = self.svd(topM, k=5)  # svd both matrices
        l2 = self.svd(attM, k=5)

        featureSpaceX = np.column_stack((l1,l2))  # stack the feature space together

        featureSpaceY = self.featureY(featureSpaceX)

        if k == None:  # recommended k for kMeans cluster
            k = int(math.sqrt(self.vcount/2))

        cluster = KMeans(n_clusters=k, random_state=0).fit_predict(featureSpaceY)
        print(cluster)
        return cluster
        #return featureSpaceX

    def cluster(self, cluster):
        '''assign each node to a set'''
        ans={}
        for index,cat in enumerate(cluster):
            if cat not in ans:
                ans[cat]=set()
            ans[cat].add(index)
        print(ans)
        return ans


    def svd(self,M,k=7):
        '''svd a given matrix, k = top k largest eigenvectors'''
        #print(M)
        u, z, v = np.linalg.svd(M, full_matrices=False)

        # z=np.diag(z)
        # v=np.dot(z,v)
        # l=np.dot(u[:,:k],v[:k,:])

        return u[:,:k]

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



# a=ANCA(s,sn,0.2,0.1)
# cluster=a.anca_calc()
# print(cluster)