import os
from glob import glob
from argparse import ArgumentParser
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys

def parse_args():
    parser = ArgumentParser(description="Run train and predict")
    parser.add_argument("-E","--hmmer_dataset", type=str, required=True,help="HMM dataset path")
    parser.add_argument("-D","--device", type=str, required=True,help="ascend device id")
    parser.add_argument("-F","--fasta_file", type=str, required=True,help="your protein fasta file, only one seq")
    parser.add_argument( "--do_train", type=bool, required=False,default=True, help="run train?")
    parser.add_argument( "--do_generate", type=bool, required=False,default=True, help="run generates?")
    parser.add_argument( "--do_cls", type=bool, required=False,default=True, help="run cls?")
    parser.add_argument("-M","--model", type=str, required=False,help="cls model name")
    args_opt = parser.parse_args()
    return args_opt

def make_mindrecord(dataset_path):
    print("================")
    print("setp 1: make mindrecord dataset")
    print("input:",dataset_path)
    print("output:",dataset_path)
    print("================")

    cmd='python ' \
        '../generate_dataset/generate_seq_for_mask.py --mask_prob 0.1 ' \
        '--data_dir '+dataset_path+' ' \
        '--vocab_file ../vocab_v2.txt ' \
        '--output_dir '+dataset_path+' ' \
        '--max_seq_length 1024 --do_train True --do_eval True --do_test True ' \
        '1> '+dataset_path+'/data_process_log.log 2> '+dataset_path+'/data_process_sys.log'
    print(cmd,flush=True)
    os.system(cmd)

def run_mask_train(dataset_path,device_id,fold):

    print("================")
    print("setp 2: run train mask model")
    print("input:",dataset_path)
    print("output:",dataset_path)
    print("================")

    cmd='python ' \
        '../ODM_mask.py ' \
        '--config_path ../config_1024.yaml ' \
        '--do_train True ' \
        '--do_eval True ' \
        '--description sequence ' \
        '--epoch_num 200 ' \
        '--early_stopping_rounds 50 ' \
        '--frozen_bert False ' \
        '--device_id '+str(device_id)+' ' \
        '--data_url '+dataset_path+' ' \
        '--load_checkpoint_url ../checkpoint_pretrain.ckpt ' \
        '--output_url '+dataset_path+' ' \
        '--task_name mask ' \
        '--train_batch_size 32 ' \

    print(cmd,flush=True)
    os.system(cmd)
    print("fold_"+str(fold)+" is running",flush=True)

def generate_mut(fasta_path,dataset_path,device_id,fold):

    fasta_name = os.path.basename(fasta_path).split(".")[0]

    model_path=dataset_path+"/mask_Best_Model.ckpt"
    csv_path="../predict_result/"+fasta_name+"/fold_"+str(fold)+"/"
    if os.path.exists(csv_path)==False:
        os.makedirs(csv_path)

    print("================")
    print("setp 3: make mask from fasta and run predict")
    print("input:",fasta_path)
    print("output:",csv_path)
    print("================")

    cmd='python ' \
        '../mpbert_mask.py ' \
        '--config_path ../config_1024.yaml ' \
        "--vocab_file  ../vocab_v2.txt " \
        '--do_predict True ' \
        '--description sequence ' \
        '--device_id '+str(device_id)+' ' \
        '--data_url '+fasta_path+' ' \
        '--load_checkpoint_url '+model_path+' ' \
        '--output_url '+csv_path+' ' \
        '--predict_mask_num 100000 ' \
        '--mask_prob 0.1 '
    print(cmd,flush=True)
    os.system(cmd)

def extract_mask_result(fasta_path,fold):
    fasta_name = os.path.basename(fasta_path).split(".")[0]
    csv_path="../predict_result/"+fasta_name+"/fold_"+str(fold)+"/"
    pickle_file="../predict_result/"+fasta_name+"/fold_0/"+fasta_name+".pkl"

    print("================")
    print("setp 4: extract predict result")
    print("input:",pickle_file)
    print("output:",csv_path+fasta_name+".csv")
    print("================")

    pickle_res=pickle.load(open(pickle_file,"rb"))

    all_df=[]

    for pred_info in tqdm(pickle_res):
        all_df.append({
            "id":pred_info["seq_id"],
            "seq":"".join(pred_info["pred_seq"]),
            "mask_logits_mean":pred_info["logits"].max(axis=1).mean(),
            "mask_logits_min":pred_info["logits"].max(axis=1).min(),
            "label":-1,
            "is_select":None
        })
    all_df=pd.DataFrame(all_df)

    #sort mask_logits_min from high to low
    all_df=all_df.sort_values(by="mask_logits_min",ascending=False)

    wild_type_info={
        "id": pickle_res[0]["seq_id"].split("_")[0]+"_WT",
        "seq": "".join(pickle_res[0]["seq"]),
        "mask_logits_mean": -1,
        "mask_logits_min":-1,
        "label": 0,
        "is_select":True
    }
    wild_type_info=pd.DataFrame([wild_type_info])
    all_df=pd.concat([wild_type_info,all_df],axis=0)
    print(all_df)
    print(all_df.values.shape)

    all_df=all_df.drop_duplicates(subset=["seq"],keep="first")
    print(all_df.values.shape)

    #select top 100000
    all_df=all_df.iloc[:100000,:].copy(deep=True)
    print(all_df.values.shape)

    #select top 100
    select_df=all_df.iloc[:201,:].copy(deep=True)
    select_df["is_select"]=[True]*201

    all_df["is_select"]=[True]*201+[False]*(len(all_df)-201)

    print(select_df)
    print(all_df)

    select_df.to_csv(csv_path+fasta_name+".csv",index=False)
    all_df.to_csv(csv_path+fasta_name+"_all.csv",index=False)
    print(len(select_df))

def do_reg(args, fasta_path, fold, device_id):
    model = args.model
    fasta_name = os.path.basename(fasta_path).split(".")[0]
    csv_path = "../predict_result/" + fasta_name + "/fold_" + str(fold) + "/"

    data_path = csv_path + fasta_name + ".csv"
    save_path = csv_path + "/" + model + "_predict/"

    print("================")
    print("step 5: run regress")
    print("input:", csv_path)
    print("output:", save_path + fasta_name + "_predict_concat_" + model + ".csv")
    print("================")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cp_data_name = save_path + fasta_name + "_fold_" + str(fold) + "." + data_path.split(".")[-1]

    print("copy", data_path, "to", cp_data_name)
    os.system("cp " + data_path + " " + cp_data_name)

    assert os.path.exists(cp_data_name)

    cmd = (
        "python ../MPB_regress.py"
        " --config_path ../config_1024.yaml"
        " --do_predict True --description classification"
        " --vocab_file ../vocab_v2.txt"
        " --device_id " + device_id +
        " --return_csv True --return_sequence False"
        " --data_url " + cp_data_name +
        " --load_checkpoint_url ../ODM_model/Models/MPB_models/MPB_Tm_model1.ckpt"
        " --output_url " + save_path
    )
    print(cmd)
    os.system(cmd)
    os.system("rm " + cp_data_name)

    return cp_data_name

def extract_reg_result(cp_data_name, args, fasta_path, fold):
    model = args.model
    fasta_name = os.path.basename(fasta_path).split(".")[0]
    csv_path = "../predict_result/" + fasta_name + "/fold_" + str(fold) + "/"
    save_path = csv_path + "/" + model + "_predict/"

    f = open(cp_data_name.split(".csv")[0] + "_predict_result.csv", "r").readlines()

    header = f[0]
    result = [header] + f[1:]

    with open(save_path + fasta_name + "_predict_result_" + model + ".csv", "w") as file:
        file.writelines(result)

    df = pd.read_csv(save_path + fasta_name + "_predict_result_" + model + ".csv")

    header_list = list(df.columns)
    header_list.remove("pred_label")
    header_list.remove("dense")

    df = df.groupby(header_list, as_index=False)['dense'].mean()
    df = df.sort_values(by="dense", ascending=False)

    df.to_csv(save_path + fasta_name + "_predict_concat_" + model + ".csv", index=False)

def main():
    args = parse_args()
    dataset_path = args.hmmer_dataset
    device_id = args.device
    fold = 0
    fasta_path = args.fasta_file

    if args.do_train:
        make_mindrecord(dataset_path)
        run_mask_train(dataset_path, device_id, fold)
    if args.run_generate:
        generate_mut(fasta_path, dataset_path, device_id,fold)
        extract_mask_result(fasta_path,fold)
    if args.do_cls:
        cp_data_name_list=do_reg(args, fasta_path, fold, device_id)
        extract_reg_result(cp_data_name_list, args, fasta_path, fold)



if __name__ == "__main__":
    main()
