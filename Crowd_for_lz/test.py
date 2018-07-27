import os
import sys
import cv2
sys.path.append('../')
import torch
import numpy as np
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

#test data and model file path

model_path = './saved_models/cmtl_Lanzhou_Crowd_96.h5'
#f_ = open('/home/yangxu/share/SDK/crowdcountingSDK/lanzhou_crowd.list', 'r')
f_ = open('/home/yangxu/share/SDK/crowdcountingSDK/video4.list', 'r')
f_lines = f_.readlines()

output_dir = './output/'
video_name = 'square_'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

output_dir = os.path.join(output_dir, 'density_maps_' + video_name + model_name)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
#load test data
#data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

net = CrowdCounter()
	  
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0
for line in f_lines:
	print line.rstrip()
	img = cv2.imread(line.rstrip(),0)
	img = cv2.resize(img, (img.shape[1], img.shape[0]))
	img = img.astype(np.float32, copy=False)
	ht = img.shape[0]
	wd = img.shape[1]
	ht_1 = (ht/4)*4
	wd_1 = (wd/4)*4
	img = cv2.resize(img,(wd_1,ht_1))
	img = img.reshape((1,1,img.shape[0],img.shape[1]))
	density_map = net(img)
	density_map = density_map.data.cpu().numpy()
	et_count = np.sum(density_map)
	print et_count
	#utils.save_density_map(density_map, output_dir + '/', 'output_' + os.path.basename(line.rstrip()).split('.')[0] + '.png')   
# mae = mae/data_loader.get_num_samples()
# mse = np.sqrt(mse/data_loader.get_num_samples())
# print 'MAE: %0.2f, MSE: %0.2f' % (mae,mse)

# f = open(file_results, 'w') 
# f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
# f.close()
