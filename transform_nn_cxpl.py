import os
import numpy as np
import PIL 
from PIL import Image
import pandas as pd

base_path = "/work/aschuette/extd_med_benchmark/results/3.0/002/classification_results"
resolution = 256
chest = True

if resolution == 512:
    directory_224 = base_path + "/224/nn_files" 
    directory_512 = base_path + "/512/nn_files"

    if not os.path.exists(directory_224):
        os.makedirs(directory_224)

    if not os.path.exists(directory_512):
        os.makedirs(directory_512)

    path_inf_img_512 = np.load(base_path + "/nn/real_inf_512.npy")
    path_nn_img_512 = np.load(base_path + "/nn/nn_images_512.npy")
    path_inf_img_224 = np.load(base_path + "/nn/real_inf_224.npy")
    path_nn_img_224 = np.load(base_path + "/nn/nn_images_224.npy")
    nn_labels = np.load(base_path + "/nn/nn_labels.npy")
    inf_labels = np.load(base_path + "/nn/real_inf_label.npy")

    n = path_inf_img_512.shape[0]
    print("Saving images for attribution maps: n=",n,flush=True)

    if chest:
        header = ["Path","No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
        "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
        "Pleural Effusion","Pleural Other","Fracture","Support Devices"]
    else:
        header = ["path","epidural","intraparenchymal","intraventricular","subarachnoid",
        "subdural","no finding"]

    paths_nn = []
    labels_nn = []
    paths_inf = []
    labels_inf = []

    for i in range(0,n):
        img_inf_512 = path_inf_img_512[i]
        img_inf_224 = path_inf_img_224[i]
        img_nn_512 = path_nn_img_512[i]
        img_nn_224 = path_nn_img_224[i]
        img_inf_512 = (255.0 / img_inf_512.max() * (img_inf_512 - img_inf_512.min())).astype(np.uint8)
        img_inf_512 = Image.fromarray(img_inf_512)
        img_inf_224 = (255.0 / img_inf_224.max() * (img_inf_224 - img_inf_224.min())).astype(np.uint8)
        img_inf_224 = Image.fromarray(img_inf_224)
        img_nn_512 = (255.0 / img_nn_512.max() * (img_nn_512 - img_nn_512.min())).astype(np.uint8)
        img_nn_512 = Image.fromarray(img_nn_512)
        img_nn_224 = (255.0 / img_nn_224.max() * (img_nn_224 - img_nn_224.min())).astype(np.uint8)
        img_nn_224 = Image.fromarray(img_nn_224)
        
        img_inf_512.save(directory_512+"/"+str(i)+'_inf.png')
        img_nn_512.save(directory_512+"/"+str(i)+'_nns.png')
        img_inf_224.save(directory_224+"/"+str(i)+'_inf.png')
        img_nn_224.save(directory_224+"/"+str(i)+'_nns.png')

        labels_nn.append(nn_labels[i])
        labels_inf.append(inf_labels[i])
        paths_nn.append(str(i)+"_nns.png")
        paths_inf.append(str(i)+"_inf.png")

    paths_nn = np.asarray(paths_nn).reshape(-1,1)
    labels_nn = np.asarray(labels_nn)
    train_data_nn = np.concatenate((paths_nn,labels_nn),axis=1)
    df_nn = pd.DataFrame(columns=header,data=train_data_nn)
    df_nn.to_csv(directory_224+"/nn_path_and_labels.csv", mode='w', header=True,index=False)
    df_nn.to_csv(directory_512+"/nn_path_and_labels.csv", mode='w', header=True,index=False)


    paths_inf = np.asarray(paths_inf).reshape(-1,1)
    labels_inf = np.asarray(labels_inf)
    train_data_inf = np.concatenate((paths_inf,labels_inf),axis=1)
    df_inf = pd.DataFrame(columns=header,data=train_data_inf)
    df_inf.to_csv(directory_224+"/inf_path_and_labels.csv", mode='w', header=True,index=False)
    df_inf.to_csv(directory_512+"/inf_path_and_labels.csv", mode='w', header=True,index=False)

    print("Done")


elif resolution == 256:
    directory_224 = base_path + "/224/nn_files" 
    directory_256 = base_path + "/256/nn_files"

    if not os.path.exists(directory_224):
        os.makedirs(directory_224)

    if not os.path.exists(directory_256):
        os.makedirs(directory_256)

    path_inf_img_256 = np.load(base_path + "/nn/real_inf_256.npy")
    path_nn_img_256 = np.load(base_path + "/nn/nn_images_256.npy")
    path_inf_img_224 = np.load(base_path + "/nn/real_inf_224.npy")
    path_nn_img_224 = np.load(base_path + "/nn/nn_images_224.npy")
    nn_labels = np.load(base_path + "/nn/nn_labels.npy")
    inf_labels = np.load(base_path + "/nn/real_inf_label.npy")

    n = path_inf_img_256.shape[0]
    print("Saving images for attribution maps: n=",n,flush=True)

    if chest:
        header = ["Path","No Finding","Enlarged Cardiomediastinum","Cardiomegaly",
        "Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
        "Pleural Effusion","Pleural Other","Fracture","Support Devices"]
    else:
        header = ["path","epidural","intraparenchymal","intraventricular","subarachnoid",
        "subdural","no finding"]

    paths_nn = []
    labels_nn = []
    paths_inf = []
    labels_inf = []

    for i in range(0,n):
        img_inf_256 = path_inf_img_256[i]
        img_inf_224 = path_inf_img_224[i]
        img_nn_256 = path_nn_img_256[i]
        img_nn_224 = path_nn_img_224[i]
        img_inf_256 = (255.0 / img_inf_256.max() * (img_inf_256 - img_inf_256.min())).astype(np.uint8)
        img_inf_256 = Image.fromarray(img_inf_256)
        img_inf_224 = (255.0 / img_inf_224.max() * (img_inf_224 - img_inf_224.min())).astype(np.uint8)
        img_inf_224 = Image.fromarray(img_inf_224)
        img_nn_256 = (255.0 / img_nn_256.max() * (img_nn_256 - img_nn_256.min())).astype(np.uint8)
        img_nn_256 = Image.fromarray(img_nn_256)
        img_nn_224 = (255.0 / img_nn_224.max() * (img_nn_224 - img_nn_224.min())).astype(np.uint8)
        img_nn_224 = Image.fromarray(img_nn_224)
        
        img_inf_256.save(directory_256+"/"+str(i)+'_inf.png')
        img_nn_256.save(directory_256+"/"+str(i)+'_nns.png')
        img_inf_224.save(directory_224+"/"+str(i)+'_inf.png')
        img_nn_224.save(directory_224+"/"+str(i)+'_nns.png')

        labels_nn.append(nn_labels[i])
        labels_inf.append(inf_labels[i])
        paths_nn.append(str(i)+"_nns.png")
        paths_inf.append(str(i)+"_inf.png")

    paths_nn = np.asarray(paths_nn).reshape(-1,1)
    labels_nn = np.asarray(labels_nn)
    train_data_nn = np.concatenate((paths_nn,labels_nn),axis=1)
    df_nn = pd.DataFrame(columns=header,data=train_data_nn)
    df_nn.to_csv(directory_224+"/nn_path_and_labels.csv", mode='w', header=True,index=False)
    df_nn.to_csv(directory_256+"/nn_path_and_labels.csv", mode='w', header=True,index=False)


    paths_inf = np.asarray(paths_inf).reshape(-1,1)
    labels_inf = np.asarray(labels_inf)
    train_data_inf = np.concatenate((paths_inf,labels_inf),axis=1)
    df_inf = pd.DataFrame(columns=header,data=train_data_inf)
    df_inf.to_csv(directory_224+"/inf_path_and_labels.csv", mode='w', header=True,index=False)
    df_inf.to_csv(directory_256+"/inf_path_and_labels.csv", mode='w', header=True,index=False)

    print("Done")   

