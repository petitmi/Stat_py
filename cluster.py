
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage,cut_tree
import pandas as pd
import matplotlib.pyplot as plt,seaborn
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
import pymysql
import config

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif']=['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
dataset=pd.read_table('/Users/petitmi/Downloads/门店分析3.txt', sep='\s+')

X_df = dataset.iloc[:, 6:].fillna(0)
X = dataset.iloc[:, 6:].fillna(0).values


#直方图
def hist_plt():
    #95%区间
    X_df[(X_df>X_df.quantile(0.025)) & (X_df<X_df.quantile(0.975))].hist(bins=100)
    plt.tight_layout()
    plt.show()

#指标相关系数
def corr():
    dataset_corr = (dataset.iloc[:,6:].corr()).round(decimals=2)
    ## 相关系数矩阵可视化
    seaborn.heatmap(dataset_corr, center=0, annot=True, cmap='YlGnBu')
    plt.show()

#数据标准化
def standatard(dataset):
    X_standard = StandardScaler().fit_transform(X)
    return X_standard

#层次聚类
def ward_heir():
    X_standard = StandardScaler().fit_transform(X)
    linked = linkage(X_standard, 'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(X_standard)

#kmeans聚类
def kmeans_al():
    X_standard = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X_standard)
    label_df = pd.DataFrame(kmeans.labels_, columns = ['label'])
    return label_df

#写入mysql
def tomysql(dataset,table):
    engine=create_engine(host=config.host,
                         user=config.user,
                         password=config.password,
                         database=config.database,
                         cursorclass=pymysql.cursors.DictCursor)
    pd.DataFrame(dataset).to_sql(table, engine, schema="mydatabase", if_exists='replace', index=True,
                                  chunksize=None, dtype=None)
#分组结果
def group_result(column):
    CLUSTER=dataset.join(label_df)
    group_label=CLUSTER.groupby(column)
    df_result_count=group_label.count().unstack()
    df_result_mean=group_label.mean().unstack()
    df_result_median=group_label.median().unstack()
    df_result_count.to_csv('/Users/petitmi/Downloads/count.csv')
    df_result_mean.to_csv('/Users/petitmi/Downloads/mean.csv')
    df_result_median.to_csv('/Users/petitmi/Downloads/median.csv')
    return df_result_count,df_result_mean,df_result_median
# print(X_df.describe())
# hist_plt()
label_df=kmeans_al()
# tomysql(label_df,'label_4')
# tomysql(dataset,"poi_week")

group_result('label')
# print(group_result('label'))