import cv2
import os
import numpy as np

f_origin = open('/home/yangxu/share/SDK/crowdcountingSDK/lanzhou_crowd.list', 'r')
o_lines = f_origin.readlines()
f_den = open('/home/yangxu/Experiments/crowdcount-cascaded-mtl/Lanzhou_Crowd/square_den.list', 'r')
d_lines = f_den.readlines()
save_dir = 'overlap/square/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

for i in range(0,len(d_lines)):
	print o_lines[i].rstrip()
	print d_lines[i].rstrip()
	origin_img=cv2.imread(o_lines[i].rstrip())
	den_img = cv2.imread(d_lines[i].rstrip())
	den_h = den_img.shape[0]
	den_w = den_img.shape[1]
	origin_img=cv2.resize(origin_img, (den_w, den_h))
	result = origin_img * 0.1 + den_img * 0.9
	#concatenate_img=np.concatenate([origin_img,den_img],axis=1)
	#idx = [i for i in range(len(name)) if name[i]==str(str(int(os.path.splitext(origin_filename)[0][-6:]))+".png")]
	#print int(os.path.splitext(origin_filename)[0][-6:])
	#print idx
	#print name[idx[0]]
	cv2.imwrite(save_dir + str(i + 1) + '.png',result)
