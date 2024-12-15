import os, argparse, time
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
import scipy.io as scio
from scipy import stats
from scipy.optimize import curve_fit
from HyperScore import HyperScore
from util.MyDataset import MyDataset
from util.loss import MSELoss
from scipy.stats import pearsonr, spearmanr, kendalltau

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  

def estimate(pred, target):
    _, _, pred = logistic_5_fitting_no_constraint(pred, target)
    plcc, _ = pearsonr(pred, target)
    srocc, _ = spearmanr(pred, target)
    krocc, _ = kendalltau(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    results = np.array([plcc, srocc, krocc, rmse])
    return results

def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=2, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--test_patch_num', help='Test patch number.', default=1, type=int)
    parser.add_argument('--lr_encoder', default=0.000002, type=float, help='learning rate in the visual encoder')
    parser.add_argument('--lr_others', default=0.0002, type=float, help='learning rate in other parts')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--data_dir', default='', type=str, help = 'path to the rendered images')
    parser.add_argument('--img_length_read', default=6, type=int, help = 'number of the using images')
    parser.add_argument('--n_ctx', default=12, type=int, help = 'number of context vectors')
    parser.add_argument('--output_dir', default='', type=str, help = 'path to the saved models')
    parser.add_argument('--save_flag', help="Flag of saving trained models", default=True, type=bool)
    parser.add_argument('--loss', default='mseloss', type=str)
    parser.add_argument('--class_token_position', default='front', type=str, help = "'middle' or 'end' or 'front'")
    parser.add_argument('--k_fold_num', default=5, type=int, help='Default 5-fold for MATE-3D')
    args = parser.parse_args()
    return args

def extend_args(args):
    args.csc = True  
    args.ctx_init = ""  
    args.prec = "fp32"  
    args.subsample_classes = "all"  

def main(args):
    print('*************************************************************************************************************************')
    cudnn.enabled = True
    save_flag = args.save_flag
    output_dir = args.output_dir
    if save_flag:
        os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    img_length_read = args.img_length_read
    test_patch_num = args.test_patch_num

    data_dir = args.data_dir
    best_all_alignment = np.zeros([args.k_fold_num, 4])
    best_all_geometry = np.zeros([args.k_fold_num, 4])
    best_all_texture = np.zeros([args.k_fold_num, 4])
    best_all_overall = np.zeros([args.k_fold_num, 4])

    for k_fold_id in range(1,args.k_fold_num + 1):

        print('The current k_fold_id is ' + str(k_fold_id)) 
        train_filename_list = 'csvfiles/train_'+str(k_fold_id)+'.csv'
        test_filename_list = 'csvfiles/test_'+str(k_fold_id)+'.csv'

        transformations_train = transforms.Compose([transforms.Resize(224),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
       
        transformations_test = transforms.Compose([transforms.Resize(224),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
        print('Trainging set: ' + train_filename_list)
        
        quality_perspectives =['alignment quality', 'geometry quality', 'texture quality', 'overall quality']   
        model = HyperScore(device, args, quality_perspectives).to(device)

    
        if args.loss == 'mseloss':
            criterion = MSELoss().to(device)
            print('Using MSE loss')
        
        clip_encoder_params = model.clip_model.visual.parameters()  
        other_params = [
            p for name, p in model.named_parameters() 
            if 'clip_model.visual' not in name and p.requires_grad
        ]  

       
        optimizer = torch.optim.AdamW([
            {'params': clip_encoder_params, 'lr': args.lr_encoder},
            {'params': other_params, 'lr': args.lr_others}  
        ], weight_decay=1e-4)
   
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        print("Ready to train network")
        print('*************************************************************************************************************************')
        best_alignment = np.zeros(4)
        best_geometry = np.zeros(4)
        best_texture = np.zeros(4)
        best_overall = np.zeros(4)
        
        min_training_loss = 10000
        
        train_dataset = MyDataset(data_dir = data_dir, datainfo_path = train_filename_list, img_length_read=img_length_read ,  transform = transformations_train, patch_num =1)
        test_dataset = MyDataset(data_dir = data_dir, datainfo_path = test_filename_list, img_length_read=img_length_read , transform = transformations_test, patch_num =1)
        
        columns = ['Epoch', 'Train_Loss', 'Train_SRCC1', 'Train_SRCC2', 'Train_SRCC3', 'Train_SRCC4',\
                'Test_SRCC1', 'Test_SRCC2', 'Test_SRCC3', 'Test_SRCC4', 'Training_time(s)']
        results_df = pd.DataFrame(columns=columns)
        
        print('Epoch\tTrain_Loss\tTrain_SRCC1\tTrain_SRCC2\tTrain_SRCC3\tTrain_SRCC4\tTest_SRCC1\tTest_SRCC2\tTest_SRCC3\tTest_SRCC4\tTraining_time(s)')
        
        for epoch in range(num_epochs):
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last = True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_patch_num , shuffle=False, num_workers=8, drop_last = True)
            n_train = len(train_loader)
            n_test = len(test_loader)
            model.train()

            start = time.time()
            batch_losses = []
            x_pre = torch.Tensor([]).cuda()
            x_gt = torch.Tensor([]).cuda()
            for i, (imgs, prompts, mos) in enumerate(train_loader):
                
                imgs = imgs.to(device)
                mos = mos.to(device)

                out = model(imgs, prompts)
                prediction = out['score_list'].squeeze()
                loss = criterion(mos, prediction) +  out['cos']

                batch_losses.append(loss.item())
                x_pre = torch.cat([x_pre,prediction], dim=0)
                x_gt = torch.cat([x_gt,mos], dim=0)
                
                optimizer.zero_grad()   
                torch.autograd.backward(loss)
                optimizer.step()
                
            x_pre_alignment = x_pre[:,0].cpu().detach().numpy()
            x_pre_geometry = x_pre[:,1].cpu().detach().numpy()
            x_pre_texture = x_pre[:,2].cpu().detach().numpy()
            x_pre_overall = x_pre[:,3].cpu().detach().numpy()
            x_gt_alignment = x_gt[:,0].cpu().detach().numpy()
            x_gt_geometry = x_gt[:,1].cpu().detach().numpy()
            x_gt_texture = x_gt[:,2].cpu().detach().numpy()
            x_gt_overall = x_gt[:,3].cpu().detach().numpy()
            
            
            train_SROCC1, _ = stats.spearmanr(x_pre_alignment, x_gt_alignment)
            train_SROCC2, _ = stats.spearmanr(x_pre_geometry, x_gt_geometry)
            train_SROCC3, _ = stats.spearmanr(x_pre_texture, x_gt_texture)
            train_SROCC4, _ = stats.spearmanr(x_pre_overall, x_gt_overall)
   
            avg_loss = sum(batch_losses) / n_train
            scheduler.step()

            end = time.time()
            train_time = end - start 

            model.eval()
            y_pre = torch.Tensor([]).cuda()
            y_gt = torch.Tensor([]).cuda()

            with torch.no_grad():
                for i, (imgs, prompts, mos) in enumerate(test_loader):
                    imgs = imgs.to(device)
                    mos = mos.to(device)
                    out = model(imgs, prompts)
                    prediction = out['score_list'].reshape(1, 4)
                    y_pre = torch.cat([y_pre,prediction], dim=0)
                    y_gt = torch.cat([y_gt,mos], dim=0)
                    
                
                y_pre_alignment = y_pre[:,0].cpu().detach().numpy()
                y_pre_geometry = y_pre[:,1].cpu().detach().numpy()
                y_pre_texture = y_pre[:,2].cpu().detach().numpy()
                y_pre_overall = y_pre[:,3].cpu().detach().numpy()
                y_gt_alignment = y_gt[:,0].cpu().detach().numpy()
                y_gt_geometry = y_gt[:,1].cpu().detach().numpy()
                y_gt_texture = y_gt[:,2].cpu().detach().numpy()
                y_gt_overall = y_gt[:,3].cpu().detach().numpy()
            

                
                test_SROCC1, _ = stats.spearmanr(y_pre_alignment, y_gt_alignment)
                test_SROCC2, _ = stats.spearmanr(y_pre_geometry, y_gt_geometry)
                test_SROCC3, _ = stats.spearmanr(y_pre_texture, y_gt_texture)
                test_SROCC4, _ = stats.spearmanr(y_pre_overall, y_gt_overall)

                results_alignment = estimate(y_pre_alignment, y_gt_alignment)
                results_geometry = estimate(y_pre_geometry, y_gt_geometry)
                results_texture = estimate(y_pre_texture, y_gt_texture)
                results_overall = estimate(y_pre_overall, y_gt_overall)

                print('%-3d\t%-8.3f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f' %
                    (epoch + 1, avg_loss, train_SROCC1, train_SROCC2, train_SROCC3, train_SROCC4, test_SROCC1, test_SROCC2, test_SROCC3, test_SROCC4, train_time))
                
                new_row = pd.DataFrame([{
                'Epoch': epoch + 1,
                'Train_Loss': avg_loss,
                'Train_SRCC1': train_SROCC1,
                'Train_SRCC2': train_SROCC2,
                'Train_SRCC3': train_SROCC3,
                'Train_SRCC4': train_SROCC4,
                'Test_SRCC1': test_SROCC1,
                'Test_SRCC2': test_SROCC2,
                'Test_SRCC3': test_SROCC3,
                'Test_SRCC4': test_SROCC4,
                'Training_time(s)': train_time
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                if avg_loss < min_training_loss:
                    if save_flag:
                        output_model_name = os.path.join(output_dir, 'model_fold' + str(k_fold_id) + '.pth')
                        torch.save(model.state_dict(), output_model_name)
                        output_mat_name = os.path.join(output_dir, 'prediction_fold' + str(k_fold_id) + '.mat')
                        scio.savemat(output_mat_name,{'y_pre':y_pre.cpu().detach().numpy(),'y_gt':y_gt.cpu().detach().numpy()})
                    best_alignment = results_alignment
                    best_geometry = results_geometry
                    best_texture = results_texture
                    best_overall = results_overall
                    min_training_loss = avg_loss

        if save_flag:
            output_excel_name =  os.path.join(output_dir, 'training_info_' + str(k_fold_id) +'.xlsx')
            results_df.to_excel(output_excel_name, index=False)
            print(f"Training results saved to {output_excel_name}")
        
        best_all_alignment[k_fold_id-1, :] = best_alignment
        best_all_geometry[k_fold_id-1, :] = best_geometry
        best_all_texture[k_fold_id-1, :] = best_texture
        best_all_overall[k_fold_id-1, :] = best_overall
       
        print("Alignment: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_alignment[0], best_alignment[1], best_alignment[2], best_alignment[3]))       
        print("Geometry: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_geometry[0], best_geometry[1], best_geometry[2], best_geometry[3]))
        print("Texture: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_texture[0], best_texture[1], best_texture[2], best_texture[3]))
        print("Overall: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_overall[0], best_overall[1], best_overall[2], best_overall[3]))
        print('*************************************************************************************************************************')
    
    best_mean_alignment = np.mean(best_all_alignment, axis=0)
    best_mean_geometry = np.mean(best_all_geometry, axis=0)
    best_mean_texture = np.mean(best_all_texture, axis=0)
    best_mean_overall = np.mean(best_all_overall, axis=0)
    print('*************************************************************************************************************************')
    print("Alignment: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_alignment[0], best_mean_alignment[1], best_mean_alignment[2], best_mean_alignment[3]))       
    print("Geometry: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_geometry[0], best_mean_geometry[1], best_mean_geometry[2], best_mean_geometry[3]))
    print("Texture: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format( best_mean_texture[0], best_mean_texture[1], best_mean_texture[2], best_mean_texture[3]))
    print("Overall: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_overall[0], best_mean_overall[1], best_mean_overall[2], best_mean_overall[3]))
    print('*************************************************************************************************************************')
    
if __name__ == "__main__":
    args = parse_args()
    extend_args(args)
    print(args)
    set_rand_seed()
    gpu = args.gpu
    with torch.cuda.device(gpu):
        main(args)