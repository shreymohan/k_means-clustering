# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:54:07 2017

@author: shrey
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import datasets
colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}

def init(df,k): #this method takes the original dataframe and the number of clusters we want and returns randomly selected centroid
    data=np.array(df)

    idx = np.random.randint(data.shape[0], size=k)

    init_centroids=data[idx,:]
    return init_centroids
     

def assignment(df_1,k,centroids,df_size): # this method finds the distance of the centroid from every data point and assigns it to the dataframe.
    


    for i in range(k):
        j=0
        s=0
        for col in range(df_size):    
            s+=(df_1.ix[:,col]-centroids[i,j])**2
            j+=1
        dist=np.sqrt(s)
        columnn='distance_from_'+str(i)
        df_1[columnn]=dist        
    
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(k)]
    df_1['cluster']=df_1.loc[:, centroid_distance_cols].idxmin(axis=1)
    df_1['cluster'] = df_1['cluster'].map(lambda x: int(x.lstrip('distance_from_')))
    df_1['color'] = df_1['cluster'].map(lambda x: colmap[x])
    return df_1
    
def update(df,df_1,k,centroids): # here the centroid is updated after distances are assigned in the dataframe
    centroids_new = copy.deepcopy(centroids)

    for num in range(k):
        c=0
        for cols in df.columns:
            centroids_new[num,c]=np.mean(df_1[df_1['cluster'] == num][cols])
            c+=1
    return centroids_new

iris = datasets.load_iris()
X = iris.data
index = ['column_'+str(i) for i in range(4)]

df=pd.DataFrame(data=X,columns=index)

df_size=len(df.columns)

dataf=df.copy(deep=True)
inc=0
centroid=init(df,k=3)  
data=assignment(df,3,centroid,df_size)
while True:
    new_cluster=data['cluster'].copy(deep=True)
    centroid=update(dataf,data,3,centroid)
    data=assignment(df,3,centroid,df_size)
    inc+=1

    if new_cluster.equals(data['cluster']):
        break

'''


inc =11 after while loop, therefore it converges after 11 iterations.


In the final dataframe 'data' after convergence:

1. 49 of the setosa flowers were clustered correctly.
2. 46 of the versicolor flowers were clustered correctly.
3. 36 of the verginica flowers were clustered correctly.
 

'''  
fig = plt.figure(figsize=(5, 5))  # visualize the clusters
plt.scatter(data['column_0'], data['column_1'], color=data['color'])
plt.show()  
'''
K means of on the 2nd dataset.
'''    

data1=pd.read_csv("/home/shrey/Desktop/ml/quiz3/SCLC_study_output_filtered_2.csv") 
data1=data1.ix[:,1:]
df_size1=len(data1.columns)

dataf1=data1.copy(deep=True)
inc=0
centroid=init(data1,k=2)  
data_new=assignment(data1,2,centroid,df_size)
while True:
    new_cluster=data['cluster'].copy(deep=True)
    centroid=update(dataf1,data_new,2,centroid)
    data_new=assignment(data1,2,centroid,df_size)
    inc+=1
    
    if new_cluster.equals(data['cluster']):
        break
    
## Visualize the clusters in the returned dataframe with any two columns 

fig = plt.figure(figsize=(5, 5))
plt.scatter(data_new['295'], data_new['464'], color=data_new['color'])
plt.show()



            
    
    
