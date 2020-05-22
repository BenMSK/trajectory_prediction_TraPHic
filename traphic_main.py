import argparse
import warnings
import re
import os
import csv
from model.model import TnpModel
from model.import_data import *
import time

warnings.filterwarnings("ignore")

'''     IMPORTANT     '''

DATASET = 'APOL'# Use Apolloscape dataset
LOG = './logs/'
LOAD = ''# Load the trained model

CUDA = True 
DEVICE = 'cuda:0'

PREDALGO = 'Traphic'

PRETRAINEPOCHS= 0# Untill pretrained epoch
TRAINEPOCHS= 200# After pretrained epoch
INPUT = 6 #Trajectory sequence input
OUTPUT = 10#Trajectory sequence output for prediction
MANUAL_SEED = 42
TENSORBOARD = True#For using tensorboard

DATA_DIR = './data/' + DATASET
MODELLOC = "./TRAPHIC_weight/"
RAW_DATA = "./data/prediction_train/"

TRAIN = True
EVAL = True

# training option 				========== Hyper-parameter ==========
BATCH_SIZE = 32
DROPOUT = 0.5
OPTIM= 'Adam'
# SGD Adam AdamW SparseAdam Adamax ASGD RMSprop Rprop 
LEARNING_RATE= 0.001
MANEUVERS = False#Ben: TODO
PRETRAIN_LOSS = 'NLL'# Negative Log-Likelihood
TRAIN_LOSS = 'MSE'
NLL_ONLY = True
WEIGHT_DECAY = 1e-4
# Trained model name for saving
NAME = '{}.{}' + '.model_{}-{}l_{}epochs.seed{}.batch{}.nll_only.{}.tar'\
			.format(INPUT, OUTPUT, PRETRAINEPOCHS + TRAINEPOCHS, MANUAL_SEED, BATCH_SIZE, NLL_ONLY)

GENERATE_DATASET = False
# If you want to generate a model for considering vehicle' only, "CLASS_TYPE = 'vehicle'",
# For any other class, there are {'bike/motor', 'human'}. For considering all class, just use 'all'
CLASS_TYPE = 'all' #(vehicle, 'bike/motor', 'human', 'all')

def apol_to_formatted(input_dir, files, output_dir, dtype):
	txtlst = []
	i = 0 
	sz = len(files)
	print("=======================================")
	for f in files:
		print("Processing {}/{} in {}...".format(i, sz, dtype))
		# print("files: ", f)
		i += 1
		splitted_name = f.split('_')
		dset_id = splitted_name[1] + splitted_name[2].zfill(2)#for prediction_train,test.zip
		
		out_name = dset_id + '.txt'
		txtlst.append(dset_id)
		
		current_time = -1
		current_frame_num = -1

		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		out = open(os.path.join(output_dir, out_name), 'w')
		f = os.path.join(input_dir, f)

		with open(f) as csv_file:
			for row in csv.reader(csv_file):
	
				each_row = row[0].split(' ')
				current_frame_num = each_row[0]
				vid_type = each_row[2]
            
				vid = int(each_row[1].split('-')[-1])
				out.write("{},{},{},{},{},{}\n".format(float(dset_id), vid, current_frame_num, each_row[3], each_row[4], vid_type))
	return txtlst

def create_data(input_dir, file_names, output_dir, dtype, threadid, class_type):
	name_lst = []
	i = 0
	sz = len(file_names)
	for f in file_names:
		print("Importing data {}/{} for {} in thread {}...".format(i, sz, dtype, threadid))
		i += 1
		dset_id = f
		
		loc = os.path.join(input_dir,dset_id+'.txt')#from 'formated folder'; i.e. formated txt file
		out = os.path.join(input_dir,dset_id+'.npy')
		import_data(loc, None, out, class_type)
		name_lst.append(out)
	
	merge(name_lst, output_dir, dtype, threadid, class_type)
	print('"merge" is finished!')



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="traphicPred command line control")

	parser.add_argument('--cuda', '-g', action='store_true', help='GPU option', default=CUDA)
	parser.add_argument('--device', '-d', help='cuda device option', default=DEVICE, type=str)

	parser.add_argument('--batch_size', '-b', help='bastch size', default=BATCH_SIZE)
	parser.add_argument('--dropout', help='dropout probability', default=DROPOUT)
	parser.add_argument('--lr', help='learning rate', default=LEARNING_RATE)
	parser.add_argument('--optim', help='optimiser', default=OPTIM)
	parser.add_argument('--w_decay', help='weight decay rate', default=WEIGHT_DECAY)
	parser.add_argument('--pretrainEpochs', '-p', help='number of epochs for pretraining', default=PRETRAINEPOCHS, type=int)
	parser.add_argument('--trainEpochs', '-e', help='number of epochs for training', default=TRAINEPOCHS, type=int)
	parser.add_argument('--maneuvers', help='maneuvers option', default=MANEUVERS, type=bool)
	parser.add_argument('--predalgo', help='prediction algorithm', default=PREDALGO)#TraPHic
	parser.add_argument('--pretrain_loss', help='pretrain loss algorithm', default=PRETRAIN_LOSS)
	parser.add_argument('--train_loss', help='train loss algorithm', default=TRAIN_LOSS)

	parser.add_argument('--dset', '-s', help='cuda device option', default=DATASET, type=str)
	parser.add_argument('--modelLoc', help='trained prediction store/load location', default=MODELLOC)
	parser.add_argument('--dir', help="location of the dataset for tracking", default=DATA_DIR)

	args = parser.parse_args()



	viewArgs = {}
	viewArgs['cuda'] = args.cuda
	viewArgs['log_dir'] = LOG
	viewArgs['batch_size'] = args.batch_size
	viewArgs['dropout'] = args.dropout
	viewArgs["lr"] = args.lr
	viewArgs["optim"] = args.optim
	viewArgs['w_decay'] = args.w_decay
	viewArgs['pretrainEpochs'] = args.pretrainEpochs
	viewArgs['trainEpochs'] = args.trainEpochs
	viewArgs["maneuvers"] = args.maneuvers
	viewArgs['predAlgo'] = args.predalgo
	viewArgs['pretrain_loss'] = args.pretrain_loss
	viewArgs['train_loss'] = args.train_loss
	viewArgs['nll_only'] = NLL_ONLY
	viewArgs['tensorboard'] = TENSORBOARD

	viewArgs['modelLoc'] = args.modelLoc
	viewArgs['dir'] = args.dir
	viewArgs['raw_dir'] = RAW_DATA
	if not args.cuda:
		args.device = 'cpu'
	viewArgs['device'] = args.device

	viewArgs['dsId'] = MANUAL_SEED# It represents seed number, which means each different seed generates different (train, val, test) set
	viewArgs['dset'] = args.dset
	viewArgs['name_temp'] = NAME
	viewArgs['input_size'] = INPUT
	viewArgs['output_size'] = OUTPUT
	viewArgs['class_type'] = CLASS_TYPE
	
	if GENERATE_DATASET:
		''' dataset ratio --> (train, val, test)==(0.7, 0.2, 0.1) '''
		np.random.seed(MANUAL_SEED)
		threadid = MANUAL_SEED
		class_type = CLASS_TYPE
		files = None

		raw_data = os.listdir(RAW_DATA)
		get_all_txtfile = [f for f in raw_data if '.txt' in f]
		dataset_cnt = len(get_all_txtfile)# Ben: Get the number of all data in 'data_dirs'
		datasets_dir = sorted(get_all_txtfile)
		np.random.shuffle(datasets_dir)
	
		datasets_for_train = datasets_dir[:int(dataset_cnt * 0.7)]
		datasets_for_val = datasets_dir[int(dataset_cnt * 0.7):int(dataset_cnt * 0.9)]
		datasets_for_test = datasets_dir[int(dataset_cnt * 0.9) :]
		
		print('dataset is generated...')
		train_loc = RAW_DATA
		output_dir = RAW_DATA + '/train/formatted/'
		files = datasets_for_train
		train_lst = apol_to_formatted(train_loc, files, output_dir, "train")
		create_data(output_dir, train_lst, args.dir, "train", threadid, class_type)

		val_loc = RAW_DATA
		output_dir = RAW_DATA + '/val/formatted/'
		files = datasets_for_val
		val_lst = apol_to_formatted(val_loc, files, output_dir, "val")
		create_data(output_dir, val_lst, args.dir, "val", threadid, class_type)

		test_loc = RAW_DATA
		output_dir = RAW_DATA + '/test_obs/formatted/'
		files = datasets_for_test
		test_lst = apol_to_formatted(test_loc, files, output_dir, "test")
		create_data(output_dir, test_lst, args.dir, "test", threadid, class_type)

		quit()
	print('using {} dataset.'.format(DATASET))

	t0 = time.time()#ben: initialize time

	model = TnpModel(viewArgs)
	if args.cuda:
		print("using cuda...\n")
	else:
		print("using cpu...\n")

	if LOAD != '':
		model.load(LOAD)
	t1 = time.time()

	if TRAIN:
		model.train(viewArgs['dsId'])

	t2 = time.time()

	if EVAL:
		model.evaluate()

	t3 = time.time()

	print('\nusing {} dataset.'.format(DATASET))
	
	print('Loading time:{}'.format(t1 - t0))
	print("Training time:{}".format(t2 - t1))
	print("Testing time:{}".format(t3 - t2))
