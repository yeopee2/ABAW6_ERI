# -*- coding: utf-8 -*-
#from ast import expr
import os
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

from networks.MTL_dan_for_RNN import resnetmtl_for_rnn
from networks.Sequentials import LSTM, GRU, Bi_LSTM, LSTM_drop, LSTM_fc, Conv_LSTM, BiLSTM_fc
from utils.utils import  PCCLoss, CCCLoss, Single_CCCLoss, Total_CCCLoss, PCC_metric, save_result_csv, save_loss_pcc_plt
from utils.dataloader import abaw_Dataset, balanced_Dataset

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/abaw5/Crop_data/', help='ABAW5 dataset path.')
    
    parser.add_argument('--DAN_ckpt', type=str, default='./DAN_bestmodel.pth', help='DAN model path.')
    parser.add_argument('--DAN_num_head', type=int, default=9, help='Number of attention head in MTL DAN.')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='RNN training epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for opt')
    parser.add_argument('--RNN_num_layers', type=int, default=2, help='RNN num layers')
    parser.add_argument('--RNN_hidden_size', type=int, default=128, help='RNN hidden size')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='LR Scheduling [ReduceLROnPlateau, StepLR, CosineAnnealingLR]')
    
    parser.add_argument('--loss_function', type=str, default='CCC', help='Loss function for RNN [PCC, CCC, MSE, MAE]')
    parser.add_argument('--model', type=str, default='LSTM', help='RNN [LSTM, GRU, Bi_LSTM, LSTM_drop, LSTM_fc, Conv_LSTM, BiLSTM_fc]')
    
    parser.add_argument('--task', type=str, default='Basic', help='[Basic, Balanced, Fold]')
    
    return parser.parse_args()



def Basic_training():

    # Data Loader
    train_dataset = abaw_Dataset(args.data_path, train= True, transform = data_transforms)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,
                              num_workers = args.workers, shuffle = False,  
                              pin_memory = True, drop_last = True)
    
    val_dataset = abaw_Dataset(args.data_path, train=False, transform = data_transforms)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size,
                            num_workers = args.workers, shuffle = False,  
                            pin_memory = True, drop_last = True)  

    
    # Training the Model
    min_val_loss = np.inf
    max_val_pcc = -1
    
    train_losses = []
    val_losses = []
    train_pccs = []
    val_pccs = []
    
    MTL_Dan.eval()
    
    for epoch in range(1, args.epochs+1):
        
        RNN.train()
        
        train_loss = 0.0
        train_pcc = 0.0
        
        print(f'Epoch {epoch}/{args.epochs}')
        for samples in tqdm(train_loader):
  
            video_name, x_train, y_label = samples
            
            optimizer.zero_grad()  #Clean the gradients

            y_label = y_label.to(device)
            mtl_output = MTL_Dan(x_train)            
            mtl_output = mtl_output.to(device)

            rnn_output = RNN(mtl_output)

            loss = criterion(rnn_output, y_label).to(device)

            loss.backward()        #Calculate gradients
            optimizer.step()       #Update weights
            
            pearson_cc = PCC_metric(video_name, rnn_output, y_label)
            
            train_loss += loss.item()
            train_pcc += pearson_cc
    
        print(f'Training Loss: {train_loss / len(train_loader)} \t PCC: {train_pcc / len(train_loader)}')
        
        
        RNN.eval()
        
        val_loss = 0.0
        val_pcc = 0.0
        
        val_video_name = []
        val_predict = []
        val_target = []       
        
        for samples in tqdm(val_loader):
            
            video_name, x_val, y_label = samples

            y_label = y_label.to(device)
            mtl_output = MTL_Dan(x_val)            
            mtl_output = mtl_output.to(device)
            
            rnn_output = RNN(mtl_output)
            
            loss = criterion(rnn_output, y_label).to(device)
            val_loss += loss.item()
            
            val_target += y_label.cpu().tolist()
            val_predict += rnn_output.cpu().tolist()
            val_video_name += video_name
            
            pearson_cc = PCC_metric(video_name, rnn_output, y_label)
            val_pcc += pearson_cc
                        
            
        print(f'Validation Loss: {val_loss / len(val_loader)} \t PCC: {val_pcc / len(val_loader)}')
        
        save_path = './checkpoints/' + mode_name
        
        if args.scheduler=='ReduceLROnPlateau': scheduler.step(val_loss)
        else: scheduler.step()
        
        
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            print('Saving The Model...')
            
            os.system('rm -rf '+save_path+'/min_loss*')
            
            torch.save(RNN.state_dict(), 
                       save_path+'/min_loss_val_PCC_%.04f'%(val_pcc / len(val_loader))+'_epoch_'+str(epoch)+'.pt')
            
            save_result_csv(mode_name, epoch, val_video_name, val_predict, val_target)

        if max_val_pcc <= val_pcc:
            max_val_pcc = val_pcc
            print('Saving The Best PCC Model...')
            
            os.system('rm -rf '+save_path+'/best_val_PCC*')
            
            torch.save(RNN.state_dict(), 
                       save_path+'/best_val_PCC_%.04f'%(val_pcc / len(val_loader))+'_epoch_'+str(epoch)+'.pt')
            
            save_result_csv(mode_name, epoch, val_video_name, val_predict, val_target)
         
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_pccs.append(train_pcc / len(train_loader))
        val_pccs.append(val_pcc / len(val_loader))

    save_loss_pcc_plt(mode_name, train_losses, train_pccs, val_losses, val_pccs)


def Balanced_training():
    
    # Data Loader
    train_dataset = balanced_Dataset(args.data_path, train= True, transform = data_transforms)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,
                              num_workers = args.workers, shuffle = False,  
                              pin_memory = True, drop_last = True)
    
    val_dataset = balanced_Dataset(args.data_path, train=False, transform = data_transforms)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size,
                            num_workers = args.workers, shuffle = False,  
                            pin_memory = True, drop_last = True)
    
    
    # Training the Model
    min_val_loss = np.inf
    max_val_pcc = -1
    
    train_losses = []
    val_losses = []
    train_pccs = []
    val_pccs = []
    
    MTL_Dan.eval()
    
    for epoch in range(1, args.epochs+1):
        
        
        RNN.train()
        
        train_loss = 0.0
        train_pcc = 0.0
        
        print(f'Epoch {epoch}/{args.epochs}')
        for samples in tqdm(train_loader):
  
            video_name, x_train, y_label = samples

            y_label = y_label.to(device)
            mtl_output = MTL_Dan(x_train)            
            mtl_output = mtl_output.to(device)

            rnn_output = RNN(mtl_output)

            loss = criterion(rnn_output, y_label).to(device)
            
            optimizer.zero_grad()  #Clean the gradients
            loss.backward()        #Calculate gradients
            optimizer.step()       #Update weights
            
            pearson_cc = PCC_metric(video_name, rnn_output, y_label)
            
            train_loss += loss.item()
            train_pcc += pearson_cc
    
        print(f'Training Loss: {train_loss / len(train_loader)} \t PCC: {train_pcc / len(train_loader)}')
        
        
        RNN.eval()
        
        val_loss = 0.0
        val_pcc = 0.0
        
        val_video_name = []
        val_predict = []
        val_target = []       
        
        for samples in tqdm(val_loader):
            
            video_name, x_val, y_label = samples

            y_label = y_label.to(device)
            mtl_output = MTL_Dan(x_val)            
            mtl_output = mtl_output.to(device)
            
            rnn_output = RNN(mtl_output)
            
            loss = criterion(rnn_output, y_label).to(device)
            val_loss += loss.item()
            
            val_target += y_label.cpu().tolist()
            val_predict += rnn_output.cpu().tolist()
            val_video_name += video_name
            
            pearson_cc = PCC_metric(video_name, rnn_output, y_label)
            val_pcc += pearson_cc
                        
            
        print(f'Validation Loss: {val_loss / len(val_loader)} \t PCC: {val_pcc / len(val_loader)}')
        
        save_path = './checkpoints/' + mode_name
        
        if args.scheduler=='ReduceLROnPlateau': scheduler.step(val_loss)
        else: scheduler.step()
        
        
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            print('Saving The Model...')
            
            os.system('rm -rf '+save_path+'/min_loss*')
            
            torch.save(RNN.state_dict(), 
                       save_path+'/min_loss_val_PCC_%.04f'%(val_pcc / len(val_loader))+'_epoch_'+str(epoch)+'.pt')
            
            save_result_csv(mode_name, epoch, val_video_name, val_predict, val_target)

        if max_val_pcc <= val_pcc:
            max_val_pcc = val_pcc
            print('Saving The Best PCC Model...')
            
            os.system('rm -rf '+save_path+'/best_val_PCC*')
            
            torch.save(RNN.state_dict(), 
                       save_path+'/best_val_PCC_%.04f'%(val_pcc / len(val_loader))+'_epoch_'+str(epoch)+'.pt')
            
            save_result_csv(mode_name, epoch, val_video_name, val_predict, val_target)
         
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_pccs.append(train_pcc / len(train_loader))
        val_pccs.append(val_pcc / len(val_loader))

    save_loss_pcc_plt(mode_name, train_losses, train_pccs, val_losses, val_pccs)


def Fold_training():
    
    # Data set
    total_dataset = abaw_Dataset(args.data_path, train= True, transform = data_transforms)

    MTL_Dan.eval()
    
    # 5 fold Training
    total_fold_num = 5
    
    kf = KFold(n_splits = total_fold_num, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(total_dataset)):
        
        train_dataset = torch.utils.data.Subset(total_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(total_dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, 
                                  num_workers = args.workers, shuffle = False, 
                                  pin_memory = True, drop_last = True)
    
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, 
                                  num_workers = args.workers, shuffle = False, 
                                  pin_memory = True, drop_last = True)
        
        min_val_loss = np.inf
        max_val_pcc = -1
        
        for epoch in range(1, args.epochs+1):
            
            RNN.train()
            
            train_loss = 0.0
            train_pcc = 0.0
            
            print(f'Fold {fold}/{total_fold_num} - Epoch {epoch}/{args.epochs}')
            
            for samples in tqdm(train_loader):
                
                video_name, x_train, y_label = samples
                
                optimizer.zero_grad()  #Clean the gradients
                
                y_label = y_label.to(device)
                mtl_output = MTL_Dan(x_train)            
                mtl_output = mtl_output.to(device)
                
                rnn_output = RNN(mtl_output)
                
                loss = criterion(rnn_output, y_label).to(device)
                
                loss.backward()        #Calculate gradients
                optimizer.step()       #Update weights
                
                pearson_cc = PCC_metric(video_name, rnn_output, y_label)
                train_loss += loss.item()
                train_pcc += pearson_cc
            
            print(f'Training Loss: {train_loss / len(train_loader)} \t PCC: {train_pcc / len(train_loader)}')
            
            
            RNN.eval()
            
            val_loss = 0.0
            val_pcc = 0.0
            
            val_video_name = []
            val_predict = []
            val_target = []  
        
            for samples in tqdm(val_loader):
                
                video_name, x_val, y_label = samples
                
                y_label = y_label.to(device)
                mtl_output = MTL_Dan(x_val)            
                mtl_output = mtl_output.to(device)
                
                rnn_output = RNN(mtl_output)

                loss = criterion(rnn_output, y_label).to(device)
                val_loss += loss.item()
                
                val_target += y_label.cpu().tolist()
                val_predict += rnn_output.cpu().tolist()
                val_video_name += video_name

                pearson_cc = PCC_metric(video_name, rnn_output, y_label)
                val_pcc += pearson_cc
                
            print(f'Validation Loss: {val_loss / len(val_loader)} \t PCC: {val_pcc / len(val_loader)}')
            
            save_path = './checkpoints/' + mode_name
        
            if args.scheduler=='ReduceLROnPlateau': scheduler.step(val_loss)
            else: scheduler.step()
            
            
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                print('Saving The Model...')

                os.system('rm -rf '+save_path+f'/fold_{fold}_min_loss*')

                torch.save(RNN.state_dict(), 
                           save_path + f'/fold_{fold}_min_loss_val_PCC_%.04f'%(val_pcc / len(val_loader))+'_epoch_'+str(epoch)+'.pt')

                
            if max_val_pcc <= val_pcc:
                max_val_pcc = val_pcc
                print('Saving The Best PCC Model...')

                os.system('rm -rf '+save_path+f'/fold_{fold}_best_val_PCC*')

                torch.save(RNN.state_dict(), 
                           save_path+f'/fold_{fold}_best_val_PCC_%.04f'%(val_pcc / len(val_loader))+'_epoch_'+str(epoch)+'.pt')




def Train_setup():
    
    # Set Models
    input_size = 22
    sequence_length = 12
    hidden_size = args.RNN_hidden_size
    num_layers = args.RNN_num_layers
    num_classes = 7
    
    # Recurrent Network
    if args.model == 'LSTM': RNN = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == 'GRU': RNN = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == 'Bi_LSTM': RNN = Bi_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == 'LSTM_drop': RNN = LSTM_drop(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == 'LSTM_fc': RNN = LSTM_fc(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == 'Conv_LSTM': RNN = Conv_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == 'BiLSTM_fc' : RNN = BiLSTM_fc(input_size, hidden_size, num_layers, num_classes).to(device)
    else: raise Exception("Please Check the RNN Model Name")
    
    # Loss function 
    if args.loss_function == 'PCC': criterion = PCCLoss()
    elif args.loss_function == 'CCC': criterion = CCCLoss()
    elif args.loss_function == 'Single_CCC' : criterion = Single_CCCLoss()
    elif args.loss_function == 'Total_CCC' : criterion =  Total_CCCLoss()
    elif args.loss_function == 'MSE': criterion = nn.MSELoss()
    elif args.loss_function == 'MAE' : criterion = nn.L1Loss()
    else: raise Exception("Please Check the Loss Function Name")
        
    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(RNN.parameters(), lr=args.lr, weight_decay = 0.001)
    
    if args.scheduler=='ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    elif args.scheduler=='StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.01)
    elif args.scheduler=='CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(15806/args.batch_size)*10, eta_min=0.001)
    else: raise Exception("Please Check the LR Scheduler Name")
    
    MTL_Dan = resnetmtl_for_rnn(num_head=args.DAN_num_head).to(device)
    
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        MTL_Dan = nn.DataParallel(MTL_Dan)
        MTL_Dan = MTL_Dan.cuda()
        RNN = nn.DataParallel(RNN)
        RNN = RNN.cuda()
    
    # Load Multitask DAN ckpt
    checkpoint = torch.load(args.DAN_ckpt)
    
    if isinstance(MTL_Dan, nn.DataParallel): # Multi GPU
        MTL_Dan.load_state_dict(checkpoint['model_state_dict'], strict=True) 
    else: # Single GPU
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict() 
        for k, v in state_dict.items(): 
            name = k[7:]
            new_state_dict[name] = v 
        MTL_Dan.load_state_dict(new_state_dict, strict = True)

    
    return MTL_Dan, RNN, criterion, optimizer, scheduler



if __name__=="__main__":
    
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    MTL_Dan, RNN, criterion, optimizer, scheduler = Train_setup()
    
    # Training
    if args.task=='Basic':
        mode_name = f'{args.model}_{args.RNN_num_layers}_{args.RNN_hidden_size}_{args.epochs}_{args.loss_function}_{args.scheduler}'
        os.makedirs('./Results/' + mode_name, exist_ok=True)
        os.makedirs('./checkpoints/' + mode_name, exist_ok=True)
        Basic_training()
    
    elif args.task=='Balanced':
        mode_name = f'balanced_{args.model}_{args.RNN_num_layers}_{args.RNN_hidden_size}_{args.epochs}_{args.loss_function}_{args.scheduler}'
        os.makedirs('./Results/' + mode_name, exist_ok=True)
        os.makedirs('./checkpoints/' + mode_name, exist_ok=True)
        Balanced_training()
        
    elif args.task=='Fold':
        mode_name = f'fold_{args.model}_{args.RNN_num_layers}_{args.RNN_hidden_size}_{args.epochs}_{args.loss_function}_{args.scheduler}'
        os.makedirs('./Results/' + mode_name, exist_ok=True)
        os.makedirs('./checkpoints/' + mode_name, exist_ok=True)
        Fold_training()
        
        
    else: raise Exception("Please Check the Task Name")

#


