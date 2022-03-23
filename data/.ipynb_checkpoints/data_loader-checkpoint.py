import h5py
import numpy as np

def load_data(a=75,b=100,index = [0,1,2,3,4,5,6,7,8,9]):
	PATH_healthy = 'data/healthy0.h5'
	PATH_patient = 'data/patients.h5'
	print("loading MRIs")
	with h5py.File(PATH_healthy, 'r') as f1:
		data_h = f1['train'][:100]

	with h5py.File(PATH_patient, 'r') as f2:
		data_p = f2['train'][:]        
	x_train =  np.concatenate([data_h[0:a],data_h[b:100], data_p[0:a],data_p[b:100]])
	x_test = np.concatenate([data_h[a:b], data_p[a:b]])
    
	print("loading table datas")
	with h5py.File('data/labels.h5', 'r') as f1:
		health = f1['x_h'][:].astype('float')
		patient = f1['x_p'][:].astype('float')
		x_test_label = np.concatenate([health[a:b], patient[a:b]])
		x_train_label = np.concatenate([health[0:a],health[b:100], patient[0:a],patient[b:100]])
		y_train_label = np.concatenate([np.zeros([75,1]), np.ones([75,1])])
		y_test_label = np.concatenate([np.zeros([25,1]), np.ones([25,1])])
	return x_train, x_test, x_test_label, x_train_label, y_test_label, y_train_label#, train_set
