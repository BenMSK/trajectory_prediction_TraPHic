# -*- coding: utf-8 -*-
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from model.Prediction.utils import outputActivation
from tensorboardX import SummaryWriter

class traphicNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(traphicNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)#Ben: TRAINING? or Just test
        self.train_flag = True

        ## Sizes of network layers
        self.dropout_prob = args['dropout_prob']
        self.encoder_size = args['encoder_size']#64
        self.decoder_size = args['decoder_size']#128
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.upp_grid_size = args['upp_grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']#32
        self.input_embedding_size = args['input_embedding_size']#32
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        self.upp_soc_embedding_size = (((args['upp_grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        self.ours = args['ours']
        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

                                                                                            # Behavioral Modification 3: Extra Inputs
        if self.ours: self.ip_emb_vel = torch.nn.Linear(2, self.input_embedding_size)#True
                                                                                            # Behavioral Modification 3: Extra Inputs
        if self.ours: self.ip_emb_nc = torch.nn.Linear(2, self.input_embedding_size)#True

        # Encoder LSTM; e_j; (input, hidden, num_of_layers)
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)# 32 x 64

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)# 64 x 128

        # Batch norm
        self.bn_conv = torch.nn.BatchNorm2d(self.encoder_size)# 64
                                                                                            #Behavioral Modification 1: Weighting the neighbors' hidden vectors after the LSTM stage
        if self.ours: self.beh_1 = torch.nn.Linear(self.encoder_size, self.encoder_size)#True; 64 x 64

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth,self.conv_3x1_depth, (3,1))
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)
        
        #TODO
        # Decoder LSTM #Ben #Ben #Ben
        if self.use_maneuvers:#False
            if self.ours:
                self.dec_lstm = torch.nn.LSTM(self.upp_soc_embedding_size + self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            else:
                self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:#True
            if self.ours:
                self.dec_lstm = torch.nn.LSTM(self.upp_soc_embedding_size + self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size,dropout = self.dropout_prob)
            else:
                self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size,dropout = self.dropout_prob)


        #batch norm
        # self.bn_dec = torch.nn.BatchNorm1d(self.decoder_size)
        #batch norm
        # self.bn_enc = torch.nn.BatchNorm1d(self.decoder_size)
        self.bnupp_soc_enc = torch.nn.BatchNorm1d(self.input_embedding_size)
        self.bn_soc_enc = torch.nn.BatchNorm1d(self.soc_embedding_size)
        self.bn_hist_enc = torch.nn.BatchNorm1d(self.upp_soc_embedding_size)
        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        #batchnorm
        self.bn_lin = torch.nn.BatchNorm1d(self.out_length)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)

        if self.ours:
            self.op_lat = torch.nn.Linear(self.upp_soc_embedding_size + self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
            self.op_lon = torch.nn.Linear(self.upp_soc_embedding_size + self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)
        else:
            self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
            self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        # self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.leaky_relu = torch.nn.ELU()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # self.bn_dec = torch.nn.BatchNorm1d(self.upp_soc_embedding_size + self.soc_embedding_size + self.dyn_embedding_size) 
        self.summary = None
        if self.args['tensorboard']:
            self.summary = SummaryWriter()

    ## Forward Pass
    def forward(self,hist, upp_nbrs, nbrs, upp_masks, masks, lat_enc, lon_enc):
        # hist = [자신의 t부터 t_h까지의 x,y , t_h-t의 1/3 지점의 x,y, t_h-t의 2/3 지점의 x,y ,
        #             t_h-t의 3/3 지점의 x,y, front, rear object의 개수]
        # upp_neighbors: 현재(t) 자신(vehId)의 x,y를 기준으로 Horizon들의 과거(t부터t_h까지)의 상대적인 dx, dy
        # neighbors: 현재(t) 자신(vehId)의 x,y를 기준으로 Neighbor들의 과거(t부터t_h까지)의 상대적인 dx, dy
        # upp_nbrs: front objects들의 history가 존재하는지에 대한 masking
        # nbrs: surrounding objects들의 history가 존재하는지에 대한 masking
        # lon_enc: [0, 1]
        # lat_enc: [0, 0, 1] --> lat, lon은 maneuver를 사용할 때 사용된다.

        ## Forward pass hist: 우리가 trajectory를 예측하고 싶어하는 대상에 대한 Layers
        ## uppnbrs --> Horizon, nbrs --> Neighbor

        if self.ours:# TODO: size of hist?
            temp = self.leaky_relu(torch.cat((self.ip_emb(hist[0:self.in_length,:,:]),self.ip_emb_vel(hist[self.in_length:,:,:])),0))#ben:ELU
            # if temp.shape[1]==self.decoder_size:
            #     temp = self.bn_enc(temp)
            _,(hist_enc,_) = self.enc_lstm(temp)
            # _,(hist_enc,_) = self.enc_lstm(self.bn_enc(self.leaky_relu(torch.cat((self.ip_emb(hist[0:15,:,:]),self.ip_emb_vel(hist[16:,:,:])),0))))
            hist_enc = self.dropout(hist_enc)
        else:
            _,(hist_enc,_) = self.enc_lstm(self.leaky_relu((self.ip_emb(hist))))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))#ELU;#Ben: Horizon map?
       
        ## Forward pass nbrs# Neighbor layer
        if self.ours:
            # self.bn1_size = nbrs.shape[1]
            # self.bn1 = torch.nn.BatchNorm1d(self.bn1_size)
            _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(torch.cat((self.ip_emb(nbrs[0:self.in_length,:,:]),self.ip_emb_vel(nbrs[self.in_length:,:,:])),0)))
            nbrs_enc = self.dropout(nbrs_enc)
        else:
            # print(nbrs)
            _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu((self.ip_emb(nbrs))))

        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        if self.ours:                                                         # Behavioral Modification 2: Adding Kinetic Flow Layer
            a = self.ip_emb(upp_nbrs[0:self.in_length,:,:])
            b = self.ip_emb_vel(upp_nbrs[self.in_length:,:,:])
            c = self.leaky_relu(torch.cat((a,b)))
            
            _, (upp_nbrs_enc,_) = self.enc_lstm(c)
            upp_nbrs_enc = self.dropout(upp_nbrs_enc)
            upp_nbrs_enc = upp_nbrs_enc.view(upp_nbrs_enc.shape[1], upp_nbrs_enc.shape[2])

                                                                                    #Behavioral Modification 1: Weighting the hidden vectors
            nbrs_enc = self.leaky_relu(self.beh_1(nbrs_enc))
            upp_nbrs_enc = self.leaky_relu(self.beh_1(upp_nbrs_enc))

        ## Masked scatter############### masking role?
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)

        if self.ours:                                                                   # Behavioral Modification 2: Adding Kinetic Flow Layer
            upp_soc_enc = torch.zeros_like(upp_masks).float()
            upp_soc_enc = upp_soc_enc.masked_scatter_(upp_masks, upp_nbrs_enc)
            upp_soc_enc = upp_soc_enc.permute(0,3,2,1)
        #

        ## Apply convolutional social pooling:maxpool
        soc_enc = self.soc_maxpool(self.leaky_relu(self.dropout(self.conv_3x1(self.bn_conv(self.leaky_relu(self.soc_conv(soc_enc)))))))
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)

                                                                               # Behavioral Modification 2: Adding Kinetic Flow Layer
        if self.ours:
            upp_soc_enc = self.soc_maxpool(self.leaky_relu(self.dropout(self.conv_3x1(self.bn_conv(self.leaky_relu(self.soc_conv(upp_soc_enc)))))))
            upp_soc_enc = upp_soc_enc.view(-1,self.upp_soc_embedding_size)

        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## Concatenate encodings:
        if self.ours:
            enc = torch.cat((self.bnupp_soc_enc(upp_soc_enc), self.bn_soc_enc(soc_enc),self.bn_hist_enc(hist_enc)),1)
        else:
            enc = torch.cat((soc_enc,hist_enc),1)


        if self.use_maneuvers:# a maneuvers is intention?
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
#            enc = self.bn_dec(enc)
            fut_pred = self.decode(enc)
            
            return fut_pred


    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        # if h_dec.shape[1]==self.decoder_size:
        #     h_dec = self.bn_dec(h_dec)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = self.bn_lin(fut_pred)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = self.dropout(fut_pred)
        fut_pred = outputActivation(fut_pred)
        return fut_pred 
