import numpy as np
import pandas as pd
import pickle
import multiprocessing
from scipy.stats import ttest_ind_from_stats
from sklearn.feature_selection import SelectFdr
from sklearn.cluster import Birch
from scipy.stats import pearsonr
import configparser
import warnings



#参数配置文件    
def get_options():
    cf = configparser.ConfigParser()
    cf.read('config.cof')
    
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
def load_dataset(case_filename,control_filename,option_dict):

    case_dataset = pd.read_table(case_filename,index_col = option_dict["index_column_name"])
    case_dataset, case_dataset_index = var_filter(case_dataset,percent = option_dict["var_percent"])

    control_dataset = pd.read_table(control_filename,index_col = option_dict["index_column_name"])
    control_dataset = control_dataset.loc[case_dataset_index,:]
    
    case_dataset_list = [case_dataset.iloc[:,index:index + option_dict["sample_for_every_periods"]] for index in range(0,option_dict["periods"] * 
            option_dict["sample_for_every_periods"],option_dict["sample_for_every_periods"])]

    control_dataset_list = [control_dataset.iloc[:,index:index + option_dict["sample_for_every_periods"]] for index in range(0,option_dict["periods"] * 
            option_dict["sample_for_every_periods"],option_dict["sample_for_every_periods"])]

    return case_dataset_list,control_dataset_list
 

#T-test 
def t_test(case,control):
    p_mean,n_mean = np.mean(case,1),np.mean(control,1)
    #print(p_mean)
    p_std,n_std = np.std(case,1),np.std(control,1)

    t_value,p_value = ttest_ind_from_stats(p_mean,p_std,case.shape[1],n_mean,n_std,control.shape[1],equal_var=False)
    p_value[np.isnan(p_value)] = 100
    
    return p_value


#根据 t 检验 然后和 fdr 的结果的过滤数据集
def fdr_t_value_filter(case,control,fdr=None):
    p_value = t_test(case,control)
    #print(p_value)
    if not fdr:
        fdr = 1

    index_mask = p_value * fdr < 0.05

    filtered_dataset = case.loc[index_mask,:]
    print("t-test filtered dataset shape: " + str(filtered_dataset.shape))
    return filtered_dataset



#准备不同时间片的数据集
def prepare(option_dict):
    case_filename, control_filename = option_dict["case_filename"],option_dict["control_filename"]
    case_dataset_list, control_dataset_list = load_dataset(case_filename,control_filename,option_dict)

    dataset_list = []
    for i in range(option_dict["periods"]):
        print("\nusing the t-test and fdr to filtered the dataset......\n")
        print("getting the {} period dataset....".format(i))
        dataset_list.append(fdr_t_value_filter(case_dataset_list[i],control_dataset_list[i],fdr = None))
    print("\n")

    return dataset_list,control_dataset_list    



#cluster the dataset, return labels and all candidate groups
def cluster_dataset(case,option_dict):
    cluster = Birch(n_clusters = option_dict["n_clusters"])
    cluster.fit(case)
    labels = cluster.labels_  #labels   

    all_condidate_clusters = []
    for i in range(option_dict["n_clusters"]):
        case_cluster = case.iloc[labels == i,:]
        #自己改进的地方   如果聚类的结果中基因的个数小于等于三个，则放弃这个类
        if case_cluster.shape[0] <= option_dict["droped_cluster_gene_count"]:
            continue 
        all_condidate_clusters.append(case_cluster)

    return all_condidate_clusters,labels


#normalize the data
def normalize(case,control):
    normalize_case = (case - control.mean().values) /  control.std().values
    return normalize_case


#calculate inner pcc
def calculate_inner_pcc(normalize_cluster):
    pearsons = []
    for gene_id_i, value_i in normalize_cluster.iterrows():
        for gene_id_j, value_j in normalize_cluster.iterrows():
            if gene_id_i == gene_id_j:
                continue
            else:
                pearson_coeffient,p_value = pearsonr(value_i,value_j)
                #print(pearson_coeffient)
                pearsons.append(pearson_coeffient)  

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


def calculate_outer_pcc(normalize_cluster,case):
    pearsons = []
    for gene_id_inner, value_inner in normalize_cluster.iterrows():
        for gene_id_outer, value_outer in case.iterrows():
            if gene_id_inner == gene_id_outer:
                continue
            else:    
                pearson_coeffient,p_value = pearsonr(value_inner,value_outer)
                #print(pearson_coeffient)
                pearsons.append(pearson_coeffient)

    average_pearsons = abs(np.mean(pearsons))
    #print("the average outer pcc: "+ str(average_pearsons))

    return average_pearsons            


#计算 CI 值  SD*PCC(inner)/PCC(outter)
def calculate_ci(case_cluster,control_cluster):
    normalize_cluster = normalize(case_cluster,control_cluster)
    #print(normalize_cluster)
    inner_pear = calculate_inner_pcc(normalize_cluster)
    outer_pear = calculate_outer_pcc(normalize_cluster,case_cluster)
    std = calculate_std(normalize_cluster)
    return inner_pear * std / outer_pear


#[(case_cluster, control_cluster),.........]
def get_cluster_pair(all_case_clusters,control_dataset):
    cluster_pair = []
    for i in range(len(all_case_clusters)):
        id_index = all_case_clusters[i].index
        #print(id_index)
        #print(control_dataset)
        control_cluster = control_dataset.loc[id_index,:]
        #print(all_case_clusters[i].head())
        #print(control_cluster.head())
        cluster_pair.append((all_case_clusters[i],control_cluster))
    return cluster_pair    


#封装一个work函数， 包含对每一个 cluster 的所有运算， 同时提供给进程池
def work(cluster_pair):
    case_cluster,control_cluster = cluster_pair
    print(case_cluster.shape)
    #print(case_cluster)
    #print(control_cluster)
    case_ci = calculate_ci(case_cluster,control_cluster)
    control_ci = calculate_ci(control_cluster,control_cluster)

    print("the case ci is: " + str(case_ci))
    print("the control ci is " + str(control_ci))
    print("")
    return case_ci,control_ci


    
#multiprocessing the cluster
def multiprocessing_clusetr(cluster_pair_list):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(work,cluster_pair_list)
    
    case_ci_list = []
    control_ci_list = []

    for item in result:
        case_ci_list.append(item[0])
        control_ci_list.append(item[1])

    return case_ci_list,control_ci_list


    
def main():
    
    option_dict = get_options()
    case_dataset_list,control_dataset_list = prepare(option_dict)
    

    for i in range(len(case_dataset_list)):
        
        all_case_clusters, case_labels = cluster_dataset(case_dataset_list[i],option_dict)
        cluster_pair_list = get_cluster_pair(all_case_clusters,control_dataset_list[i])

        case_ci_list,control_ci_list = multiprocessing_clusetr(cluster_pair_list)
        print("\n")
        print("the {} of time period: \n".format(i))
        print("case result: " + str(case_ci_list) + "\n")
        print("control result: " + str(control_ci_list) + "\n")
                
     
if __name__ == '__main__':
    main()
    


       

