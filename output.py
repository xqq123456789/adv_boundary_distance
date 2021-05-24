import sys
import matplotlib.pyplot as plt
from PIL import Image
import os

filePath="/home/xqq/5-7/5-17/result/resnet_lin_eps_0.031" # 文件夹路径
fileList=os.listdir(filePath)

for file in fileList:
    #filename=open(os.path.join(filePath,file))
    print(file) # 文件名
    print('-------------------------')
    ori_result=[] 
    adv_result=[]
    flag=0
    with open(filePath+'/'+file,'r') as f:
        for line in f:
            if line[0]=='s':
                flag=0         
            if line[0]=='a':
                flag=1
            if line[0]!='s' and line[0]!='a' and line[0]!='o' and flag==0:
                ori_result.append(float(line))
            if line[0]!='s' and line[0]!='a' and line[0]!='o' and flag==1:
                adv_result.append(float(line))
    plt.hist(ori_result, bins=40,density=0,  facecolor="blue", edgecolor="black",alpha=0.5,stacked=True) 
    #plt.hist(result_list, facecolor="blue", edgecolor="black") 
    plt.xlabel("Distance")
    plt.ylabel("Number")
    #plt.savefig("picture/5_7_FGSM_resultv1_400.png") 
    #plt.close()
    plt.hist(adv_result, bins=38,density=0,  facecolor="yellow", edgecolor="black",alpha=0.5,stacked=True)
    #plt.hist(result_list_adv, facecolor="yellow", edgecolor="black")  
    plt.xlabel("Distance")
    plt.ylabel("Number")
    plt.savefig("/home/xqq/5-7/5-17/result/picture/resnet_lin_eps_0.031"+file[:-4]+".png") 
    plt.close()

