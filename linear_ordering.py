import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA


path = "...\Linear_and_cluster\sad_data.xlsx"

def get_data(path: str):

    df = pd.read_excel(path)
    df.set_index('Country', inplace=True)
    df = df.rename_axis(None)

    return df

def standarized(data_frame):
    stand_df = data_frame.copy()
    col_names = stand_df.columns
    scaler = StandardScaler()
    stand_df[col_names] = scaler.fit_transform(data_frame[col_names])

    return stand_df

def hellwig_method(data_frame):

    hellwig_df = data_frame.copy()
    col_names = hellwig_df.columns
    
    max_values = list(np.max(hellwig_df, axis=0))

    for i,j in enumerate(col_names):
        hellwig_df[j] = pow(hellwig_df[j] - max_values[i],2)

    hellwig_df['di'] = hellwig_df[col_names].sum(axis=1)

    d0 = np.mean(hellwig_df['di']) + 2*  np.std(hellwig_df['di'])

    hellwig_df['si'] = 1 - hellwig_df['di']/d0

    hellwig_df = hellwig_df.sort_values(by='si', ascending=False)
    hellwig_df['rank_h'] = range(1, len(hellwig_df) + 1)

    return hellwig_df

def standarized_sums(data_frame):
    stand_sum_df = data_frame.copy()
    stand_sum_df['si'] = stand_sum_df[stand_sum_df.columns].sum(axis=1)

    stand_sum_df['s_sum'] = (stand_sum_df['si'] - min(stand_sum_df['si'])) / max (stand_sum_df['si'] - min(stand_sum_df['si']))
    stand_sum_df = stand_sum_df.sort_values(by='s_sum', ascending=False)
    stand_sum_df['rank_s'] = range(1, len(stand_sum_df) + 1)

    return stand_sum_df


def main_func(data_file: str):
    df = get_data(data_file)
    cluster_df = df.copy(deep=True)

  
    '''
    corr = df.corr()
    sns.set(font_scale=0.9)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Heatmap - Matrix of Correlation")
    '''


    mean_values = list(np.mean(df, axis=0))
    std_values = list(np.std(df, axis=0))

  
    cv_values = list()
    for i in range(len(mean_values)):
        cv_values.append(std_values[i]/mean_values[i])


    df['Crime_rate'] = -1 * df['Crime_rate']
    df['Cost_of_living'] = -1 * df['Cost_of_living']
    df['Emissions_of_co2'] = -1 * df['Emissions_of_co2']


    stand_df = standarized(df)


    hellwig_df = hellwig_method(stand_df)
    stand_sum_df = standarized_sums(stand_df)

    
    stand_sum_df['rank_h'] = hellwig_df['rank_h']

    data_kmean = stand_df
    num_cluster = 3

    pca = PCA(2)
    data_pca = pca.fit_transform(data_kmean)
    

    kmeans = KMeans(num_cluster)
    label = kmeans.fit_predict(data_pca)
    

    

    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    
  
    
    for i in u_labels:
        plt.scatter(data_pca[label == i , 0] , data_pca[label == i , 1] , label = i)

    for i, country in enumerate(data_kmean.index):
        plt.annotate(country, (data_pca[i, 0], data_pca[i, 1]))

    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()

main_func(path)

