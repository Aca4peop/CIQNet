# five-fold split for ODV JVQD and 360SVQD
import numpy as np
import scipy.io as sio

class ODVSource():
    def __init__(self):
        self.__files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds=self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.__files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        fold={}
        train_index = np.concatenate((fold_indexs[0],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[2],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/ODV_dmos_sorted.mat')
        name = mat['videos']
        for i in range(540):
            tmp=name[i]
            tmp = tmp.replace(' ', '')
            tmp=tmp.replace('.yuv', '.mp4')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/ODV_dmos_sorted.mat')
        dmos = mat['dmos']
        dmos = np.array(dmos).flatten()/100.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.__files[i])
        return image_new

class JVQDSource():
    def __init__(self):
        self.files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds = self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        fold = {}
        train_index = np.concatenate((fold_indexs[0], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[2],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/JVQD2.mat')
        name = mat['videos']
        for i in range(60):
            tmp=name[i]
            tmp = tmp.replace(' ', '').replace('\'', '')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/JVQD2.mat')
        dmos = mat['Jmos']
        dmos = np.array(dmos).flatten()/100.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.files[i])
        return image_new

class SVQDSource():
    def __init__(self):
        self.files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds = self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        # fold 1
        fold = {}
        train_index = np.concatenate((fold_indexs[0], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[2],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/360SVQD.mat')
        name = mat['name']
        for i in range(64):
            tmp=name[i]
            tmp = tmp+'.mkv'
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/360SVQD.mat')
        dmos = mat['mos']
        dmos = np.array(dmos).flatten()/5.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.files[i])
        return image_new

class ODISource():
    def __init__(self):
        self.train_mos = []
        self.test_mos = []
        self.train_names = []
        self.test_names = []
        self.__feed_set('CVIQ')
        self.train_mos=np.concatenate(self.train_mos)
        self.test_mos=np.concatenate(self.test_mos)

    def __train_validate(self,dmos,files):
        assert len(dmos)==len(files)
        num_videos=len(files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        test_index = index[1::6]
        train_index = np.delete(index, test_index)
        np.random.shuffle(train_index)
        return train_index,test_index


    def __get_dmos(self,indicator):
        mat = sio.loadmat('./datasets/ODI_enhance.mat')
        mos = mat[indicator]
        mos = np.array(mos).flatten()
        name = mat['index_'+indicator]
        name = np.array(name).flatten()
        return mos,name
    def __gen_file_name(self,indicator:str,name):
        namebox = []
        for n in name:
            namebox.append(indicator + '/%d.png' % (n))
        return namebox

    def __feed_set(self,indicator:str):

        mos, files = self.__get_dmos(indicator)
        train_index, test_index=self.__train_validate(mos,files)
        self.train_mos.append(mos[train_index])
        self.test_mos.append(mos[test_index])
        self.train_names=self.train_names+self.__gen_file_name(indicator,files[train_index])
        self.test_names=self.test_names+self.__gen_file_name(indicator,files[test_index])