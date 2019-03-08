import pickle
import os
import sys
import numpy as np
#import pc_util
#import scene_util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
NUM_CLASSES = 19
class ScannetDataset():

	def __init__(self, root, npoints=8192, split='train'):
		self.npoints = npoints
		self.root = root
		self.split = split
		'''
		self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
		with open(self.data_filename,'rb') as fp:
			self.scene_points_list = pickle.load(fp)
			self.semantic_labels_list = pickle.load(fp)

			for i in range(len(self.semantic_labels_list)):
				idx = np.where(self.semantic_labels_list[i] > 18)
				self.semantic_labels_list[i][idx] = 0
			#self.xxx = pickle.load(fp)
		'''
		ALL_FILES_TRAIN = provider.getDataFiles(os.path.join(root, 'all_files.txt'))
		self.scene_points_list = []
		self.semantic_labels_list = []
		for h5_filename in ALL_FILES_TRAIN:
			scene_points, semantic_labels = provider.loadDataFile(os.path.join(root, h5_filename))
			self.scene_points_list.append(scene_points[:, :, 0:3])
			self.semantic_labels_list.append(semantic_labels)
		self.scene_points_list = np.concatenate(self.scene_points_list, 0)
		self.semantic_labels_list = np.concatenate(self.semantic_labels_list, 0)

		if split=='train':
			labelweights = np.zeros(NUM_CLASSES)
			for seg in self.semantic_labels_list:
				tmp, _ = np.histogram(seg, range(NUM_CLASSES+1))
				labelweights += tmp
			labelweights = labelweights.astype(np.float32)
			labelweights = labelweights/np.sum(labelweights)
			self.labelweights = 1/np.log(1.2+labelweights)
		elif split =='test':
			self.labelweights = np.ones(NUM_CLASSES)

	def __getitem__(self, index):
		point_set = self.scene_points_list[index]
		semantic_seg = self.semantic_labels_list[index].astype(np.int32) # 4096 only one batch
		sample_weight = np.ones(len(semantic_seg))

		coordmax = np.max(point_set, axis=0)
		coordmin = np.min(point_set, axis=0)
		# smpmin = np.maximum(coordmax-[1.5, 1.5, 3.0], coordmin)
		# smpmin[2] = coordmin[2]
		# smpsz = np.minimum(coordmax-smpmin, [1.5, 1.5, 3.0])
		# smpsz[2] = coordmax[2]-coordmin[2]
		isvalid = False
		for i in range(1):
			curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :] #choice a number within the range(4096) equivalent with randomly choice a points as center
			curmin = curcenter-[10, 10, 1.5]#[0.75, 0.75, 1.5]20x20 square
			curmax = curcenter+[10, 10, 1.5]#[0.75, 0.75, 1.5]
			curmin[2] = coordmin[2] # replace the curmin z with the global minimum z
			curmax[2] = coordmax[2] # replace the curman z with the global maximum z
			curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),  axis=1)==3
			#curchoice = [True]*len(semantic_seg)
			cur_point_set = point_set[curchoice, :]
			cur_semantic_seg = semantic_seg[curchoice]
			if len(cur_semantic_seg)==0:
				continue
			mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
			vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
			vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
			isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
			if isvalid:
				break
		choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
		point_set = cur_point_set[choice,:]
		semantic_seg = cur_semantic_seg[choice]
		mask = mask[choice]
		sample_weight = self.labelweights[semantic_seg]
		sample_weight *= mask
		#sample_weight = np.ones(len(semantic_seg))

		return point_set, semantic_seg, sample_weight

	def __len__(self):
		return len(self.scene_points_list)


class ScannetDatasetWholeScene():

	def __init__(self, root, npoints=8192, split='train'):
		self.npoints = npoints
		self.root = root
		self.split = split
		self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
		with open(self.data_filename, 'rb') as fp:
			self.scene_points_list = pickle.load(fp)
			self.semantic_labels_list = pickle.load(fp)
		if split=='train':
			labelweights = np.zeros(21)
			for seg in self.semantic_labels_list:
				tmp,_ = np.histogram(seg,range(22))
				labelweights += tmp
			labelweights = labelweights.astype(np.float32)
			labelweights = labelweights/np.sum(labelweights)
			self.labelweights = 1/np.log(1.2+labelweights)
		elif split=='test':
			self.labelweights = np.ones(21)

	def __getitem__(self, index):
		point_set_ini = self.scene_points_list[index]
		semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
		coordmax = np.max(point_set_ini, axis=0)
		coordmin = np.min(point_set_ini, axis=0)
		nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
		nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
		point_sets = list()
		semantic_segs = list()
		sample_weights = list()
		isvalid = False
		for i in range(nsubvolume_x):
			for j in range(nsubvolume_y):
				curmin = coordmin+[i*1.5,j*1.5,0]
				curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
				curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
				cur_point_set = point_set_ini[curchoice,:]
				cur_semantic_seg = semantic_seg_ini[curchoice]
				if len(cur_semantic_seg)==0:
					continue
				mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
				choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
				point_set = cur_point_set[choice,:] # Nx3
				semantic_seg = cur_semantic_seg[choice] # N
				mask = mask[choice]
				if sum(mask)/float(len(mask))<0.01:
					continue
				sample_weight = self.labelweights[semantic_seg]
				sample_weight *= mask # N
				point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
				semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
				sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
		point_sets = np.concatenate(tuple(point_sets),axis=0)
		semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
		sample_weights = np.concatenate(tuple(sample_weights),axis=0)
		return point_sets, semantic_segs, sample_weights

	def __len__(self):
		return len(self.scene_points_list)

# class ScannetDatasetVirtualScan():
# 	def __init__(self, root, npoints=8192, split='train'):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
#         with open(self.data_filename,'rb') as fp:
#             self.scene_points_list = pickle.load(fp)
#             self.semantic_labels_list = pickle.load(fp)
# 	if split=='train':
# 	    labelweights = np.zeros(21)
# 	    for seg in self.semantic_labels_list:
# 		tmp,_ = np.histogram(seg,range(22))
# 		labelweights += tmp
# 	    labelweights = labelweights.astype(np.float32)
# 	    labelweights = labelweights/np.sum(labelweights)
# 	    self.labelweights = 1/np.log(1.2+labelweights)
# 	elif split=='test':
# 	    self.labelweights = np.ones(21)
#     def __getitem__(self, index):
#         point_set_ini = self.scene_points_list[index]
#         semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
# 	sample_weight_ini = self.labelweights[semantic_seg_ini]
# 	point_sets = list()
# 	semantic_segs = list()
# 	sample_weights = list()
# 	for i in range(8):
# 	    smpidx = scene_util.virtual_scan(point_set_ini,mode=i)
# 	    if len(smpidx)<300:
# 			continue
#             point_set = point_set_ini[smpidx,:]
# 	    semantic_seg = semantic_seg_ini[smpidx]
# 	    sample_weight = sample_weight_ini[smpidx]
# 	    choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
# 	    point_set = point_set[choice,:] # Nx3
# 	    semantic_seg = semantic_seg[choice] # N
# 	    sample_weight = sample_weight[choice] # N
# 	    point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
# 	    semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
# 	    sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
# 	point_sets = np.concatenate(tuple(point_sets),axis=0)
# 	semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
# 	sample_weights = np.concatenate(tuple(sample_weights),axis=0)
#         return point_sets, semantic_segs, sample_weights
#     def __len__(self):
#         return len(self.scene_points_list)

if __name__=='__main__':
	pass
	# d = ScannetDatasetWholeScene(root = './data', split='test', npoints=8192)
	# labelweights_vox = np.zeros(21)
	# for ii in range(len(d)):
	# 	print(ii)
	# 	ps,seg,smpw = d[ii]
	# 	for b in range(ps.shape[0]):
	# 		_, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b,smpw[b,:]>0,:], seg[b,smpw[b,:]>0], res=0.02)
	# 		tmp,_ = np.histogram(uvlabel,range(22))
	# 		labelweights_vox += tmp
	# print(labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32)))
	# exit()


