import numpy as np
import pandas as pd
import pickle
import multiprocessing
from scipy.stats import ttest_ind_from_stats
from sklearn.feature_selection import SelectFdr
from sklearn.cluster import Birch
from scipy.stats import pearsonr
import configparser
import pickle
from itertools import combinations



#参数配置文件    
def get_options():
    cf = configparser.ConfigParser()
    cf.read('config_single.cof')
    
    option_dict = dict()
    for key,value in cf.items("Main"):
        option_dict[key] = eval(value)

    return option_dict

#方差过滤
def var_filter(dataset,percent = 0.8):
    new_index_series = dataset.var(1)
    print("the row dataset shape is: " + str(dataset.shape))
    dataset = dataset.loc[new_index_series != 0,:]

    new_index = new_index_series[new_index_series != 0].sort_values().index
    new_dataset = dataset.reindex(index = new_index)
    
    if percent > 0 and percent < 1:
        new_dataset = new_dataset.iloc[:int(new_dataset.shape[0]*percent),:]
    else:
        print("the percent is error!")    

    print("the var filtered dataset shape: " + str(new_dataset.shape))
    
    return new_dataset,new_dataset.index

#加载数据集
def get_dataset(option_dict):
    case_filename = option_dict["case_filename"]
    case_dataset = pd.read_table(case_filename,index_col = option_dict["index_column_name"])
    
    case_dataset, case_dataset_index = var_filter(case_dataset,percent = option_dict["var_percent"])
    case_dataset_list = [case_dataset.iloc[:,index:index + option_dict["sample_for_every_periods"]] for index in range(0,option_dict["periods"] * 
            option_dict["sample_for_every_periods"],option_dict["sample_for_every_periods"])]

    return case_dataset_list
 

#cluster the dataset, return labels and all candidate groups
def cluster_dataset(case,option_dict):
    cluster = Birch(n_clusters = option_dict["n_clusters"])
    cluster.fit(case)
    labels = cluster.labels_  #labels   

    all_condidate_clusters = []
    for i in range(option_dict["n_clusters"]):
        case_cluster = case.iloc[labels == i,:]
        #自己改进的地方   如果聚类的结果中基因的个数小于等于n个，则放弃这个类
        if case_cluster.shape[0] <= option_dict["droped_cluster_gene_count"]:
            continue 
        all_condidate_clusters.append(case_cluster)

    return all_condidate_clusters,labels


#calculate inner pcc
def calculate_inner_pcc(normalize_cluster):
    gene_id = normalize_cluster.index.tolist()
    gene_combinations = list(combinations(gene_id, 2))

    pearsons = [pearsonr(normalize_cluster.loc[tuple_[0],:], \
        normalize_cluster.loc[tuple_[1],:])[0] for tuple_ in gene_combinations]
    
    pearsons = [number for number in pearsons if str(number) != "nan"]
    average_pearsons = abs(np.mean(pearsons))

    #print("the average inner pcc: " + str(average_pearsons))
    return average_pearsons


def calculate_std(normalize_cluster):
    stds = []
    for gene_id,value in normalize_cluster.iterrows():
        std = value.std()
        #print(std)
        stds.append(std)
    average_stds = np.mean(stds)    
    #print("the average std is: "+ str(average_stds))    
    return average_stds   


def calculate_outer_pcc(normalize_cluster,period_cluster):
    pearsons = []

    for gene_id_inner, value_inner in normalize_cluster.iterrows():
        for gene_id_outer, value_outer in period_cluster.iterrows():
            if gene_id_inner == gene_id_outer:
                continue
            else:    
                pearson_coeffient,p_value = pearsonr(value_inner,value_outer)
                #print(pearson_coeffient)
                pearsons.append(pearson_coeffient)

    pearsons = [number for number in pearsons if str(number) != "nan"]
    average_pearsons = abs(np.mean(pearsons))

    #print("the average outer pcc: " + str(average_pearsons))
    return average_pearsons            


#计算 CI 值  SD*PCC(inner)/PCC(outter)
def calculate_ci(normalize_cluster,period_cluster):
     #print(normalize_cluster)
    inner_pear = calculate_inner_pcc(normalize_cluster)
    outer_pear = calculate_outer_pcc(normalize_cluster,period_cluster)
    std = calculate_std(normalize_cluster)
    return inner_pear * std / outer_pear



#封装一个work函数， 包含对每一个 cluster 在不同的时间片的值的所有运算， 同时提供给进程池
def work(cluster_pair):
    cluster_list,case_dataset_list = cluster_pair
    case_cis = []
    for index,cluster in enumerate(cluster_list):
        case_cis.append(calculate_ci(cluster,case_dataset_list[index]))
    print(case_cis)    
    return case_cis


def get_cluster_pair(normalize_cluster_list,case_dataset_list):
    cluster_pair = [(cluster,case_dataset_list) for cluster in normalize_cluster_list]
    return cluster_pair   
    
#multiprocessing the cluster
def multiprocessing_clusetr(all_periods_case_clusters,case_dataset_list):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    cluster_pair = get_cluster_pair(all_periods_case_clusters,case_dataset_list)
    result = pool.map(work,cluster_pair)

    return result


# 得到每个cluster 在不同时间片的 cluster
def get_all_periods_clusters(all_case_clusters,case_dataset_list):
    
    #print(len(all_case_clusters))
    result_cluster = []
    for cluster in all_case_clusters:
        tmp = []
        cluster_index = cluster.index
        for raw_cluster in case_dataset_list:
            tmp.append(raw_cluster.loc[cluster_index,:])   
        result_cluster.append(tmp)
    
    return  result_cluster
            
        

  
def main():
    
    option_dict = get_options()
    #准备不同时间片的数据
    case_dataset_list = get_dataset(option_dict)
    
    for i in range(len(case_dataset_list)):
        
        #对单个时间片进行聚类,得到该时间片内的所有group
        all_case_clusters, case_labels = cluster_dataset(case_dataset_list[i],option_dict)
        
        #得到每个cluster在不同的时间片内的所有group[(1,2,3,4,5),(1,2,3,4,5)]
        all_periods_case_clusters = get_all_periods_clusters(all_case_clusters, case_dataset_list)

        #result = work((all_periods_case_clusters[0],case_dataset_list))
        #exit()

        result = multiprocessing_clusetr(all_periods_case_clusters,case_dataset_list)

        print("\n")
        print("the {} of time period: \n".format(i))
        print("result: " + str(result) + "\n")

        with open("{}_period_result_single.pkl".format(i),"wb") as f:
            pickle.dump(result,f)

        exit()    


                
     
if __name__ == '__main__':
    main()
    


       

