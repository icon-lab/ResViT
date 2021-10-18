import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode !='aligned_mat' and opt.dataset_mode !='unaligned_mat':
        if opt.dataset_mode == 'aligned':
            from data.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()
        elif opt.dataset_mode == 'unaligned':
            from data.unaligned_dataset import UnalignedDataset
            dataset = UnalignedDataset()
        elif opt.dataset_mode == 'single':
            from data.single_dataset import SingleDataset
            dataset = SingleDataset()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)  
            print("dataset [%s] was created" % (dataset.name()))
        dataset.initialize(opt)
         
    #custom data loader    
    if opt.dataset_mode == 'aligned_mat' or opt.dataset_mode == 'unaligned_mat':   
        #data location
        target_file=opt.dataroot+'/'+opt.phase+'/7_slice.mat'
        f = h5py.File(target_file,'r') 
        slices=np.array(f['data_x']).shape[3]/2
        samples=range(np.array(f['data_x']).shape[2])
        #if (not opt.serial_batches):
         #   random.shuffle(samples)
        if opt.which_direction=='AtoB':
            data_x=np.array(f['data_x'])[:,:,samples,slices-opt.input_nc/2:slices+opt.input_nc/2+1]
            data_y=np.array(f['data_y'])[:,:,samples,slices-opt.output_nc/2:slices+opt.output_nc/2+1]
        else:            
            data_y=np.array(f['data_y'])[:,:,samples,slices-opt.input_nc/2:slices+opt.input_nc/2+1]
            data_x=np.array(f['data_x'])[:,:,samples,slices-opt.output_nc/2:slices+opt.output_nc/2+1]
        if opt.dataset_mode == 'unaligned_mat':  
            if opt.isTrain and opt.phase!='val':
                print("Training phase")
                random.shuffle(samples)
            else:
                print("Testing phase")
            data_y=data_y[:,:,samples,:]
        data_x=np.transpose(data_x,(3,2,0,1))
        data_y=np.transpose(data_y,(3,2,0,1))
        if  len(data_x.shape)<4:
            data_x=np.expand_dims(data_x,0)
        if  len(data_y.shape)<4:
            data_y=np.expand_dims(data_y,0)            
        dataset=[]
        for train_sample in range(data_x.shape[1]):
            data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
            data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
            dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'B':torch.from_numpy(data_y[:,train_sample,:,:]), 
            'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
        print('#training images = %d' % train_sample)
        print(data_x.shape)
        print(data_y.shape)        
    #else:
    #    raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    return dataset 



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
