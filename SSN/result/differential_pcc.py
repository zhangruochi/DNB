import pickle

with open("reference_result.pkl","rb") as f:
    reference_result_list = pickle.load(f)


with open("SSN_result.pkl","rb") as f:
    SSN_result_list = pickle.load(f)

differential_pcc_result = []   
for i,time_clusters in enumerate(SSN_result_list):
    tmp_result = []
    for j,cluster in enumerate(time_clusters):
        tmp_result.append([abs(num1-num2) for num1,num2 in zip(cluster,reference_result_list[i][j])])

    differential_pcc_result.append(tmp_result)    

with open("differential_result.txt","w") as f:
    f.write(str(differential_pcc_result))
    



     