import re
import os
import subprocess
import torch
import argparse


from model.Prediction.traphic import traphicNet
from model.Prediction.social import highwayNet
from model.Prediction.utils import ngsimDataset
from model.Prediction.traphicEngine import TraphicEngine
from model.Prediction.socialEngine import SocialEngine
from torch.utils.data import DataLoader
import datetime




class TnpModel:

    def __init__(self, inArgs):
        torch.manual_seed(inArgs['dsId'])#seed becomes 'dataset Id'
        torch.cuda.manual_seed(inArgs['dsId'])

        self.args = {}
        self.args["batch_size"] = inArgs["batch_size"]
        self.args["pretrainEpochs"] = inArgs["pretrainEpochs"]
        self.args["trainEpochs"] = inArgs["trainEpochs"]
        self.args['cuda'] = inArgs["cuda"]
        self.args['device'] = inArgs['device']
        self.args['modelLoc'] = inArgs['modelLoc']#Ben: the location of 'trained model'
        self.args["optim"] = inArgs["optim"]

        # Network Arguments
        self.args['dropout_prob'] = inArgs["dropout"]
        self.args['encoder_size'] = 64
        self.args['decoder_size'] = 128
        self.args['in_length'] = inArgs['input_size']#Ben: INPUT (history length)
        self.args['out_length'] = inArgs['output_size']#Ben: OUTPUT (output seq length)
        self.args['grid_size'] = (13,3)
        self.args['upp_grid_size'] = (7,3)
        self.args['soc_conv_depth'] = 64
        self.args['conv_3x1_depth'] = 16
        self.args['dyn_embedding_size'] = 32
        self.args['input_embedding_size'] = 32
        self.args['num_lat_classes'] = 3
        self.args['num_lon_classes'] = 2
        self.args['use_maneuvers'] = inArgs["maneuvers"]
        self.args['ours'] = (inArgs["predAlgo"] == "Traphic")
        self.args['nll_only'] = inArgs['nll_only']
        self.args["learning_rate"] = inArgs["lr"]
        self.args["predAlgo"] = inArgs["predAlgo"]#TraPHic
        self.args["w_decay"] = inArgs['w_decay']
        
        # currentDT = datetime.datetime.now()
        # self.args['name'] = "{}_{}_model.tar".format(inArgs["predAlgo"], currentDT.strftime("%Y_%m_%d_%H_%M"))
        self.args['name'] = inArgs['name_temp'].format(self.args["predAlgo"], inArgs['dset'])
        self.args["pretrain_loss"] = inArgs['pretrain_loss']
        self.args['train_loss'] = inArgs['train_loss']
        self.args['dir'] = inArgs['dir']
        self.args['raw_dir'] = inArgs['raw_dir']
        self.args['dsId'] = inArgs['dsId']
        self.args['log_dir'] = inArgs['log_dir']
        self.args['tensorboard'] = inArgs['tensorboard']
        self.args['class_type'] = inArgs['class_type']
        if self.args["predAlgo"] == "Traphic":# Ben: Declare the network
            self.net = traphicNet(self.args)
        else:
            self.net = highwayNet(self.args)

        if self.args['cuda']:
            self.net = self.net.cuda(self.args['device'])
        


    def eval_one(self, dsId=None):
        if dsId:
            self.args['dsId'] = dsId

        self.net.train_flag = False

        self.net.eval()
        d = os.path.join(self.args['modelLoc'], self.args['name'])

        if os.path.exists(d):
            self.net.load_state_dict(torch.load(d))
            print("\n[INFO]: model {} loaded".format(d))
        else:
            print("\n[INFO]: can not find model at {} to evaluate, using existing net".format(d))

        if self.args["cuda"]:
            self.net.cuda(self.args['device'])



        if self.args["optim"] == "Adam":
            optim = torch.optim.Adam(self.net.parameters(),lr=self.args['learning_rate'],weight_decay=self.args["w_decay"])
        elif self.args["optim"] == "SGD":
            optim = torch.optim.SGD(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "AdamW":
            optim = torch.optim.AdamW(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "SparseAdam":
            optim = torch.optim.SparseAdam(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "Adamax":
            optim = torch.optim.Adamax(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "ASGD":
            optim = torch.optim.ASGD(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "Rprop":
            optim = torch.optim.Rprop(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "RMSprop":
            optim = torch.optim.RMSprop(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "LBFGS":
            optim = torch.optim.LBFGS(self.net.parameters(),lr=self.args['learning_rate'])
        else:
            print("undefined optimizer.")
            return

        crossEnt = torch.nn.BCELoss()
        self.net()

    def load(self, d=None, load=False):
        self.net.eval()#Ben: model.eval() will notify all your layers that you are in eval mode, 
                       #     that way, batchnorm or dropout layers will work in eval mode instead of training mode.
        if not d:#Ben: Get a location(path)
            d = os.path.join(self.args['modelLoc'], self.args['name'])
        else:
            if load:
                self.args['name'] = d 
            d = os.path.join(self.args['modelLoc'], d)
        
        if os.path.exists(d):
            self.net.load_state_dict(torch.load(d))
            print("\n[INFO]: model {} loaded\n".format(d))
        else:
            print("\n[INFO]: can not find model at {} to evaluate, using existing net".format(d))


    def train(self, dsId=None):
        if dsId:
            self.args['dsId'] = dsId
    
        self.net.train_flag = True
        self.net.train()
        if self.args["cuda"]:
            self.net.cuda(self.args['device'])

        if self.args["optim"] == "Adam":
            optim = torch.optim.Adam(self.net.parameters(),lr=self.args['learning_rate'],weight_decay=self.args["w_decay"])
        elif self.args["optim"] == "SGD":
            optim = torch.optim.SGD(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "AdamW":
            optim = torch.optim.AdamW(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "SparseAdam":
            optim = torch.optim.SparseAdam(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "Adamax":
            optim = torch.optim.Adamax(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "ASGD":
            optim = torch.optim.ASGD(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "Rprop":
            optim = torch.optim.Rprop(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "RMSprop":
            optim = torch.optim.RMSprop(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "LBFGS":
            optim = torch.optim.LBFGS(self.net.parameters(),lr=self.args['learning_rate'])
        else:
            print("undefined optimizer.")
            return

        crossEnt = torch.nn.BCELoss()#ben: Binary Cross Entrophy

        print('loading data in {}...'.format(self.args['dsId']))
        trSet_path = os.path.join(self.args["dir"], "trainSet")
        valSet_path = os.path.join(self.args["dir"], "valSet")
        
        trSet = ngsimDataset(trSet_path, self.args["dir"], self.args["raw_dir"], 'train', self.args['dsId'], self.args['class_type'], t_h=self.args['in_length'], t_f=self.args['out_length'])
        valSet = ngsimDataset(valSet_path, self.args["dir"], self.args["raw_dir"], 'val', self.args['dsId'], self.args['class_type'], t_h=self.args['in_length'], t_f=self.args['out_length'])

        trDataloader = DataLoader(trSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=trSet.collate_fn)
        # trDataloader = DataLoader(valSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=valSet.collate_fn)
        valDataloader = DataLoader(valSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=valSet.collate_fn)

        print('start training {}...'.format(self.args["predAlgo"]))
        if self.args["predAlgo"] == "Traphic":
            engine = TraphicEngine(self.net, optim, trDataloader, valDataloader, self.args)
        else:
            engine = SocialEngine(self.net, optim, trDataloader, valDataloader, self.args)

        engine.start()


        

    def evaluate(self, dsId=None):
        if dsId:# Ben: dataset id?? TODO
            self.args['dsId'] = dsId

        self.net.train_flag = False
        self.net.eval()# Ben: Ready for evaluation
        d = os.path.join(self.args['modelLoc'], self.args['name'])

        if os.path.exists(d):
            self.net.load_state_dict(torch.load(d, map_location = 'cuda:0'))#Ben: error handled
            print("\n[INFO]: model {} loaded".format(d))
        else:
            print("\n[INFO]: can not find model at {} to evaluate, using existing net".format(d))

        if self.args["cuda"]:
            self.net.cuda(self.args['device'])



        if self.args["optim"] == "Adam":
            optim = torch.optim.Adam(self.net.parameters(),lr=self.args['learning_rate'],weight_decay=self.args["w_decay"])
        elif self.args["optim"] == "SGD":
            optim = torch.optim.SGD(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "AdamW":
            optim = torch.optim.AdamW(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "SparseAdam":
            optim = torch.optim.SparseAdam(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "Adamax":
            optim = torch.optim.Adamax(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "ASGD":
            optim = torch.optim.ASGD(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "Rprop":
            optim = torch.optim.Rprop(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "RMSprop":
            optim = torch.optim.RMSprop(self.net.parameters(),lr=self.args['learning_rate'])
        elif self.args["optim"] == "LBFGS":
            optim = torch.optim.LBFGS(self.net.parameters(),lr=self.args['learning_rate'])
        else:
            print("undefined optimizer.")
            return

        crossEnt = torch.nn.BCELoss()

        print('loading data in {}...'.format(self.args['dsId']))
        trSet_path = os.path.join(self.args["dir"], "trainSet")
        valSet_path = os.path.join(self.args["dir"], "valSet")
        tstSet_path = os.path.join(self.args["dir"], "testSet")

        trSet = ngsimDataset(trSet_path, self.args["dir"], self.args["raw_dir"], 'train', self.args['dsId'], t_h=self.args['in_length'], t_f=self.args['out_length'])
        trDataloader = DataLoader(trSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=trSet.collate_fn)

        testSet = ngsimDataset(valSet_path, self.args["dir"], self.args["raw_dir"], 'val', self.args['dsId'], t_h=self.args['in_length'], t_f=self.args['out_length'])
        testDataloader = DataLoader(testSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=testSet.collate_fn)

        valSet = ngsimDataset(tstSet_path, self.args["dir"], self.args["raw_dir"], 'val', self.args['dsId'], t_h=self.args['in_length'], t_f=self.args['out_length'])
        valDataloader = DataLoader(valSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=valSet.collate_fn)

        print('start testing {}...'.format(self.args["predAlgo"]))
        if self.args["predAlgo"] == "Traphic":
            engine = TraphicEngine(self.net, optim, trDataloader, valDataloader, self.args)
        else:
            engine = SocialEngine(self.net, optim, trDataloader, valDataloader, self.args)

        engine.eval(trDataloader)
        # engine.eval(testDataloader)

    def result_viz(self):#TODO
        # This function is for visualizing the network output and ground truth trajectory
        # print('loading data in {}...'.format(self.args['dsId']))
        tstSet_path = os.path.join(self.args["dir"], "testSet")

        testSet = ngsimDataset(valSet_path, self.args["dir"], self.args["raw_dir"], 'val', self.args['dsId'], t_h=self.args['in_length'], t_f=self.args['out_length'])
        testDataloader = DataLoader(testSet,batch_size=self.args['batch_size'],shuffle=True,num_workers=4,collate_fn=testSet.collate_fn)
        if self.args["predAlgo"] == "Traphic":
            engine = TraphicEngine(self.net, optim, trDataloader, valDataloader, self.args)
        else:
            engine = SocialEngine(self.net, optim, trDataloader, valDataloader, self.args)
        
