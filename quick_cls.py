import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils import data
import torch
import pandas as pd
import numpy as np
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import sys
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def get_labels_dict(labels_set={}): # 如果为空则读取，否则保存
    tag2idx = {}
    if len(labels_set) > 0:
        with open(os.path.join(MODEL_SAVE, "labels_dict.txt"), "w") as f:
            for idx, label in enumerate(labels_set):
                f.write("{}:{}\n".format(idx, label))
                tag2idx[label] = int(idx)
    else:
        with open(os.path.join(MODEL_SAVE,"labels_dict.txt")) as f_idx2tag: # 格式为index:tag 样本数
            for line in f_idx2tag:
                (idx, label) = line.strip("\n").split(":")
                tag2idx[label] = int(idx)
    return tag2idx
def train_save():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FILE)
    config = AutoConfig.from_pretrained(CONFIG_FILE)
    
    LT = torch.LongTensor
    train_input_ids, train_attn_masks, train_token_type_ids, train_labels = [],[],[],[]
    val_input_ids, val_attn_masks, val_token_type_ids, val_labels = [],[],[],[]

    print("config.hidden_size, config.embedding_size, config.max_length: ")
    print(config.hidden_size, config.embedding_size, config.max_length)
    
    train_file = os.path.join(DATA_DIR,"train.csv") # 默认名称
    val_file = os.path.join(DATA_DIR,"val.csv")
    train_data = pd.read_csv(train_file,header=0).dropna() # 默认with headers，数据格式为text,label
    val_data = pd.read_csv(val_file,header=0).dropna()
    
    # 保证val与train中类别数量相同
    assert set(train_data.iloc[:,1]) == set(val_data.iloc[:,1])
    tag2idx = get_labels_dict(labels_set = set(train_data.iloc[:,1]))

    for index, row in train_data.iterrows():
        itext = row[0]
        tokenized_itext = tokenizer(itext.strip(), max_length=MAX_LENGTH, padding='max_length', truncation=True)
        train_input_ids.append(tokenized_itext['input_ids'])
        train_attn_masks.append(tokenized_itext['attention_mask'])
        train_token_type_ids.append(tokenized_itext['token_type_ids'])
        train_labels.append(tag2idx[row[1]])
    for index, row in val_data.iterrows():
        itext = row[0]
        tokenized_itext = tokenizer(itext.strip(), max_length=MAX_LENGTH, padding='max_length', truncation=True)
        val_input_ids.append(tokenized_itext['input_ids'])
        val_attn_masks.append(tokenized_itext['attention_mask'])
        val_token_type_ids.append(tokenized_itext['token_type_ids'])
        val_labels.append(tag2idx[row[1]])
    
    train_dataset = TensorDataset(LT(train_input_ids),  LT(train_attn_masks), LT(train_token_type_ids), LT(train_labels))
    val_dataset = TensorDataset(LT(val_input_ids),  LT(val_attn_masks), LT(val_token_type_ids), LT(val_labels))

    assert set(i[3].item() for i in train_dataset) == set(i[3].item() for i in val_dataset) == set(tag2idx.values())

    num_labels = len(set(tag2idx.values()))
    print("num_labels: ", num_labels)
    print(len(train_dataset), len(val_dataset))

    model_cls = AutoModelForSequenceClassification.from_pretrained(PRETRAINED, num_labels=num_labels)
   
    train_dataloader = data.DataLoader(train_dataset,batch_size=BATCH_SIZE)
    val_dataloader = data.DataLoader(val_dataset,batch_size=BATCH_SIZE)

    # optimizer = torch.optim.SGD(model_cls.parameters(),lr=0.02,momentum=0.5,weight_decay=4e-4)
    optimizer = AdamW(model_cls.parameters(), lr=LR)
      
    num_training_steps = len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    model_cls.to(DEVICE)
    progress_bar = tqdm(range(num_training_steps))
    print("num_training_steps: ",num_training_steps)
    # Model training and val
    for epoch in range(EPOCHS):
        loss_sum=0.0
        accu=0
        train_pred = []
        train_label = []
        val_pred = []
        val_label = []
        train_num_batch = 0      
        val_num_batch = 0  
        model_cls.train()
        for batch in train_dataloader:
            batch = {"input_ids":batch[0].to(DEVICE),
                     "attention_mask":batch[1].to(DEVICE),
                     "token_type_ids":batch[2].to(DEVICE),
                     "labels":batch[3].to(DEVICE)}
            # print({k:v.shape for k,v in batch.items()})
            # sys.exit()
            outputs = model_cls(**batch)
            loss = outputs.loss
            loss.backward() #Back propagation
            optimizer.step() #Gradient update
            lr_scheduler.step()
            optimizer.zero_grad()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            loss_sum+=loss.cpu()
            accu+=accuracy_score(batch['labels'].cpu(),predictions.cpu())
            train_pred.extend(predictions.cpu()) 
            train_label.extend(batch['labels'].cpu())  
            progress_bar.update(1)
            train_num_batch+=1
        val_loss_sum=0.0
        val_accu=0
        model_cls.eval()
        for batch in val_dataloader:
            batch = {"input_ids":batch[0].to(DEVICE),
                     "attention_mask":batch[1].to(DEVICE),
                     "token_type_ids":batch[2].to(DEVICE),
                     "labels":batch[3].to(DEVICE)}
            with torch.no_grad():
                outputs=model_cls(**batch)
                loss = outputs.loss
                val_loss_sum+=loss.cpu()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                val_accu+=accuracy_score(batch['labels'].cpu(),predictions.cpu())
                val_pred.extend(predictions.cpu())                  
                val_label.extend(batch['labels'].cpu())
            val_num_batch+=1
        print("epoch %d,train loss:%f,train acc:%f,val loss:%f,val acc:%f" % (epoch+1,loss_sum/len(train_dataset),accu/train_num_batch,val_loss_sum/len(val_dataset),val_accu/val_num_batch))
        # f1 = metrics.f1_score(tags_true, tags_pred, average='weighted')
        # precision = metrics.precision_score(tags_true, tags_pred, average='weighted')
        # recall = metrics.recall_score(tags_true, tags_pred, average='weighted')
        # accuracy = metrics.accuracy_score(tags_true, tags_pred)    
        if epoch%10 == 0 and epoch>0:
            output_model_file = os.path.join(MODEL_SAVE, MODEL_SAVE_NAME)
            # output_vocab_file = os.path.join(MODEL_SAVE, 'vocab.txt')
            torch.save(model_cls, output_model_file)
            print("model saved")
            print("train data: ")
            print(confusion_matrix(train_label, train_pred))
            print("val data: ")
            print(confusion_matrix(val_label, val_pred))
            print("train data: ")
            print(classification_report(train_label, train_pred, labels=list(tag2idx.values()), target_names=list(tag2idx.keys())))
            print("val data: ")
            print(classification_report(val_label, val_pred, labels=list(tag2idx.values()), target_names=list(tag2idx.keys())))
def test():
    model_file = os.path.join(MODEL_SAVE, MODEL_SAVE_NAME)
    model = torch.load(model_file).to('cpu')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FILE)
    tag2idx = get_labels_dict()
    idx2tag = {v:k for k,v in tag2idx.items()}
    model.eval()
    test_file = os.path.join(DATA_DIR, "test.csv")
    df_test = pd.read_csv(test_file, header=0)
    with open(os.path.join(DATA_DIR, SAVE_TEST_RESULT), "w") as f:
        for text, label in df_test.values:
            tokenized_text = tokenizer(text, max_length=MAX_LENGTH, padding='max_length', truncation=True)
            batch = {"input_ids":tokenized_text["input_ids"].to(DEVICE),
                     "attention_mask":tokenized_text["attention_mask"].to(DEVICE),
                     "token_type_ids":tokenized_text["token_type_ids"].to(DEVICE),
                     "labels":tag2idx[label].to(DEVICE)}
            outputs = model(**batch)  
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)        
            f.write("{}@@@@@@{}@@@@@@{}\n".format(text, label, idx2tag[predictions.item()]))
def test_single():
    model_file = os.path.join(MODEL_SAVE, MODEL_SAVE_NAME)
    model = torch.load(model_file).to('cpu')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FILE)
    tag2idx = get_labels_dict()
    idx2tag = {v:k for k,v in tag2idx.items()}
    model.eval()
    while True:
        print("Enter your sentence..")
        text = input()
        tokenized_text = tokenizer.encode(text, max_length=MAX_LENGTH, padding='max_length', truncation=True)
        batch = {"input_ids":tokenized_text["input_ids"].to(DEVICE),
                 "attention_mask":tokenized_text["attention_mask"].to(DEVICE),
                 "token_type_ids":tokenized_text["token_type_ids"].to(DEVICE),
                 "labels":tag2idx[label].to(DEVICE)}
        outputs = model(**batch)  
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)    
        print("分类结果： ", idx2tag[predictions.item()])

if __name__ == '__main__':
    TOKENIZER_FILE = '/data_ml/juan.du/main_prod/bert_models/bert_base_chinese'
    PRETRAINED = '/data_ml/juan.du/main_prod/bert_models/albert_chinese_tiny'  #Use small version of Albert
    CONFIG_FILE = '/data_ml/juan.du/main_prod/bert_models/albert_chinese_tiny'
    DATA_DIR = './data/cls/'
    SAVE_TEST_RESULT = 'test_result.csv'
    MODEL_SAVE = './models/cls/'
    MODEL_SAVE_NAME = 'pytorch_albert_qishun_cat.bin'
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
    MAX_LENGTH = 50
    EPOCHS = 20
    LR = 5e-4
    BATCH_SIZE = 32
     
    if len(sys.argv) > 1:
        action_type = sys.argv[1]
    else:
        raise Exception("Provide action type: train, test, test_single")
    if action_type == 'train':
        train_save()
    elif action_type == 'test':
        test()
    elif action_type == 'test_single':
        test_single()
