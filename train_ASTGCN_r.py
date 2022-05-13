#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import os.path as osp
from time import time
import shutil
import argparse
import configparser
from model.ASTGCN_r import make_model
from lib.utils import load_graphdata_channel1, load_custom_graphdata, compute_val_loss_mstgcn, predict_and_save_results_mstgcn, get_logger, init_log
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse


result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/METR_LA_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

######################### PARSE DATA ARGUMENT #########################

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
begin_year = int(data_config['begin_year'])
end_year = int(data_config['end_year'])


######################### PARSE MODEL ARGUMENT #########################


model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('./experiments', dataset_name, folder_dir)
print('params_path:', params_path)


######################### GET LOGGER #########################
# run_id = 'astgcn_lr_%g_bs_%d_%s/' % (
#                 learning_rate, batch_size,
#                 time.strftime('%m%d%H%M%S'))
# log_dir = os.path.join('./log/', run_id)

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# logger = get_logger('./log/', __name__, str(self.year) + 'info.log')
logger = init_log()


########################################## LOAD DATA ##########################################
# train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
#     graph_signal_matrix_filename, num_of_hours,
#     num_of_days, num_of_weeks, DEVICE, batch_size)

# train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor = load_custom_graphdata(
#     graph_signal_matrix_filename, 2011, DEVICE, batch_size)


########################################## LOAD GRAPH ##########################################
# adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

# adj_mx = np.load(osp.join(adj_filename, str(2011)+"_adj.npz"))["x"]
# num_of_vertices = int(adj_mx.shape[0])


########################################## MODEL DEFINITION ##########################################
# net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
#                  num_for_predict, len_input, num_of_vertices)







def train_main(year):
    global result

    logger.info('START TRAINING YEAR {} !'.format(year))

    # Dataset Definition
    adj_mx = np.load(osp.join(adj_filename, str(year)+"_adj.npz"))["x"]
    num_of_vertices = int(adj_mx.shape[0])
    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor = load_custom_graphdata(
        graph_signal_matrix_filename, str(year), DEVICE, batch_size)

    # Model Definition

    if (start_epoch == 0) and (not os.path.exists(str(params_path) + '/' + str(year) + '/')):
        os.makedirs(str(params_path) + '/' + str(year) + '/')
        logger.info('create params directory %s' % (str(params_path) + '/' + str(year) + '/'))
    elif (start_epoch == 0) and (os.path.exists(str(params_path) + '/' + str(year) + '/')):
        shutil.rmtree(str(params_path) + '/' + str(year) + '/')
        os.makedirs(str(params_path) + '/' + str(year) + '/')
        logger.info('delete the old one and create params directory %s' % (str(params_path) + '/' + str(year) + '/'))
    elif (start_epoch > 0) and (os.path.exists(str(params_path) + '/' + str(year) + '/')):
        logger.info('train from params directory %s' % (str(params_path) + '/' + str(year) + '/'))
    else:
        raise SystemExit('Wrong type of model!')

    net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                num_for_predict, len_input, num_of_vertices)

    # Logging
    log_every = 5
    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function=='masked_mse':
        criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
        masked_flag=1
    elif loss_function=='masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter('runs/' + 'log_dir')
    # print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{}'.format(total_param))

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    if start_epoch > 0:

        params_filename = os.path.join(params_path, str(year),'epoch_%s.tar' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        logger.info('start epoch:', start_epoch)

        logger.info('load weight from: {}'.format(params_filename))

    if year > begin_year :
        
        epo_list = []
        params_path_prev_year = params_path + '/{year}'.format(year = int(year) - 1)    # load the previous year model as the initial model
        for filename in os.listdir(params_path_prev_year): 
            if filename.endswith(".tar"):
                epo_list.append(filename[6:]) 					# already has .tar in it
        epo_list= sorted(epo_list)
        params_filename = '{}/epoch_{epo_num}'.format(params_path_prev_year, epo_num = epo_list[-1])
        assert os.path.exists(params_filename), 'Weights at {} not found'.format(params_filename)

        checkpoint = torch.load(params_filename)
        # print(checkpoint['model_state_dict'])

        pretrained_dict = checkpoint['model_state_dict']
        model_dict = net.state_dict()
        pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() } # Keep the layer that has the same dimension 

        model_dict.update(pretrained_dict)

        net.load_state_dict(model_dict)
        logger.info('load weight from: {}'.format(params_filename))

    # train model

    use_time = []
    total_time = 0
    wait = 0
    start_time = time()
    patience = 50

    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, str(year),'epoch_%s.tar' % epoch)        # resume training if start_epoch != 0

        val_loss = 0
        training_loss = 0

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels,missing_value)
            else :
                loss = criterion(outputs, labels)


            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            # if global_step % 1000 == 0:

            #     print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

        end_time = time()
        total_time += (end_time - start_time)
        use_time.append((end_time - start_time))

        ####################### EVALUATE IN TRAINING #######################

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)

        if val_loss < best_val_loss:
            wait = 0
            best_val_loss = val_loss
            best_epoch = epoch
            if not os.path.exists(str(params_path) + '/' +  str(year) + '/'):
                os.makedirs(str(params_path) + '/' +  str(year) + '/')
            torch.save({'model_state_dict': net.state_dict()}, params_filename)
            # save_model
            print('save parameters to file: %s' % params_filename)
        elif val_loss >= best_val_loss:
            wait += 1
            if wait == patience:
                logger.warning('Early stopping at epoch: %d' % epoch)
                break

        if (epoch % log_every) == log_every - 1:
            message = 'Epoch [{}/{}]  train_loss: {:.4f}, val_loss: {:.4f}, ' \
                        '{:.1f}s'.format(epoch, epochs,
                                        np.mean(training_loss), val_loss, float(use_time[-1]))
            logger.info(message)


        
    message = 'YEAR : {} \n Total training time is : {:.1f}s \n Average traning time is : {:.1f}s'.format(year, total_time, sum(use_time)/len(use_time))
    logger.info(message)		
    result[year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}

    logger.info('best epoch: {}'.format(best_epoch))

    # apply the best model on the test set

    params_filename = os.path.join(params_path, str(year), 'epoch_%s.tar' % best_epoch)
    print('load weight from:', params_filename)
    net.load_state_dict(torch.load(params_filename)['model_state_dict'])
    predict_main(best_epoch, test_loader, test_target_tensor,metric_method , 'test', net, year)


def predict_main(global_step, data_loader, data_target_tensor,metric_method , type, net, year):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    # params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    # print('load weight from:', params_filename)

    # net.load_state_dict(torch.load(params_filename))

    logger.info("[*] year {}, testing".format(year))

    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, params_path, type, year, result)



def main():
    for year in range(begin_year, end_year + 1):

        logger.info("[*] Year {} load from {}_30day.npz".format(year, osp.join(graph_signal_matrix_filename, str(year)))) 

        train_main(year)

    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            for year in range(begin_year, end_year+1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            info+="{:.2f}\t".format(result[i][j][year])
            logger.info("{}\t{}\t".format(i,j) + info)

    for year in range(begin_year, end_year+1):
        if year in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
            logger.info(info)


if __name__ == "__main__":
    main()
    # predict_main(13, test_loader, test_target_tensor,metric_method, _mean, _std, 'test')














