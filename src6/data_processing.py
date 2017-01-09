import os
import cv2
import numpy as np
from lxml import etree
import random
from keras.utils import np_utils


class DataProcesser:
    def __init__(self):
        self.idx2vid_dev, self.idx2vid_val = {}, {}
        self.vid2idx_dev, self.vid2idx_val = {}, {}
        self.idx2name_ref, self.idx2name_query = {}, {}
        self.vid2vehicle_info = {}

    def equalize_hist_all(self, root='../data/val/'):
        raw_root, out_root = root + 'images/', root + 'normalized/'
        if not os.path.exists(out_root):
            os.mkdir(out_root)
        cnt = 0
        for parent, _, files in os.walk(raw_root):
            for name in files:
                img = cv2.imread(parent + name)
                b, g, r = cv2.split(img)
                bb, gg, rr = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
                [row, col] = b.shape

                if row > col:
                    d = row - col
                    add_block = np.zeros((d, row))
                    new_bb = np.vstack((bb.T, add_block))
                    new_gg = np.vstack((gg.T, add_block))
                    new_rr = np.vstack((rr.T, add_block))
                    new_bb = new_bb.T
                    new_gg = new_gg.T
                    new_rr = new_rr.T
                else:
                    d = col - row
                    add_block = np.zeros((d, col))
                    new_bb = np.vstack((add_block, bb))
                    new_gg = np.vstack((add_block, gg))
                    new_rr = np.vstack((add_block, rr))

                new_bb, new_gg, new_rr = np.uint8(new_bb), np.uint8(new_gg), np.uint8(new_rr)
                new_image = cv2.merge([new_bb, new_gg, new_rr])

                res = cv2.resize(new_image, (100, 100), interpolation=cv2.INTER_CUBIC)
                new_name = out_root + name
                cv2.imwrite(new_name, res)
                cnt += 1
                if cnt % 500 == 0:
                    print 'Processed', cnt, 'images!'

    def load_data(self, validation_split=0.2, root='../data/'):
        # load training data
        vehicle_dct = {}
        xml_file = os.path.join(root, 'train/train_gt.xml')
        print xml_file
        xml_root = etree.parse(xml_file).getroot()
        img_names = []
        for item in xml_root.getiterator('Item'):
            attr = item.attrib
            cid, img_name = int(attr['colorID']), attr['imageName']
            vid, mid = int(attr['vehicleID']), int(attr['modelID'])
            img_names.append(img_name)
            vehicle_dct[img_name] = (cid, vid, mid)
            self.vid2vehicle_info[vid] = (cid, mid)

        # shuffle and split training set
        random.shuffle(img_names)
        val_size = int(validation_split * len(vehicle_dct)) if type(validation_split) == float else validation_split
        dev_size = len(img_names) - val_size

        # input image dimensions
        img_rows, img_cols, img_channel = 100, 100, 3
        X_dev = np.zeros((dev_size, img_channel, img_rows, img_cols), dtype='float32')
        print X_dev.shape
        tmp = np.zeros((img_channel, img_rows, img_cols), dtype='float32')
        train_root = os.path.join(root, 'train/normalized')
        # load develop set
        for i, img_name in enumerate(img_names[:-val_size]):
            img = cv2.imread(os.path.join(train_root, img_name + '.jpg'))
            img = img.astype('float32') / 255.
            tmp[0], tmp[1], tmp[2] = img[:, :, 0],img[:, :, 1], img[:, :, 2]
            X_dev[i] = tmp
            vid = vehicle_dct[img_name][1]
            self.idx2vid_dev[i] = vid
            try:
                self.vid2idx_dev[vid].append((img_name, i))
            except:
                self.vid2idx_dev[vid] = [(img_name, i)]

        # load validation set
        # only keep images appeared in the develop set
        val_data = filter(lambda name: vehicle_dct[name][1] in self.vid2idx_dev, img_names[-val_size:])
        X_val = np.zeros((len(val_data), img_channel, img_rows, img_cols), dtype='float32')
        for i, img_name in enumerate(val_data):
            img = cv2.imread(os.path.join(train_root, img_name + '.jpg'))
            img = img.astype('float32') / 255.
            tmp[0], tmp[1], tmp[2] = img[:, :, 0],img[:, :, 1], img[:, :, 2]
            X_val[i] = tmp
            vid = vehicle_dct[img_name][1]
            self.idx2vid_val[i] = vid
            try:
                self.vid2idx_val[vid].append((img_name, i))
            except:
                self.vid2idx_val[vid] = [(img_name, i)]

        n_unique_cars = len(self.vid2idx_dev)
        print 'Unique cars:', n_unique_cars
        # y_dev[i]: car indices (in X_dev) of the same VID with X_dev[i]
        # y_val: car indices (in **!X_dev!**) of the same VID with X_val[i]
        y_dev = [self.vid2idx_dev[self.idx2vid_dev[idx]] for idx in xrange(len(X_dev))]
        y_val = [self.vid2idx_dev[self.idx2vid_val[idx]] for idx in xrange(len(X_val))]
        return (X_dev, y_dev), (X_val, y_val), n_unique_cars

    def load_test_data(self, root='../data/'):
        # load test data
        xml_file = os.path.join(root, 'val/val_list.xml')
        print xml_file
        xml_root = etree.parse(xml_file).getroot()
        ref_images, query_images = [], []
        for item in xml_root.getiterator('Items'):
            attr = item.attrib
            if attr['name'] == 'ref':
                ref_images = map(lambda node: node.attrib['imageName'], item.getchildren())
            else:
                query_images = map(lambda node: node.attrib['imageName'], item.getchildren())

        # input image dimensions
        img_rows, img_cols, img_channel = 100, 100, 3
        X_ref = np.zeros((len(ref_images), img_channel, img_rows, img_cols), dtype='float32')
        X_query = np.zeros((len(query_images), img_channel, img_rows, img_cols), dtype='float32')
        print X_ref.shape, X_query.shape

        tmp = np.zeros((img_channel, img_rows, img_cols), dtype='float32')
        val_root = os.path.join(root, 'val/normalized')
        for i, img_name in enumerate(ref_images):
            img = cv2.imread(os.path.join(val_root, img_name + '.jpg'))
            img = img.astype('float32') / 255.
            # tmp[0], tmp[1], tmp[2] = cv2.split(img)
            tmp[0], tmp[1], tmp[2] = img[:, :, 0],img[:, :, 1], img[:, :, 2]
            X_ref[i] = tmp
            self.idx2name_ref[i] = img_name

        for i, img_name in enumerate(query_images):
            img = cv2.imread(os.path.join(val_root, img_name + '.jpg'))
            img = img.astype('float32') / 255.
            # tmp[0], tmp[1], tmp[2] = cv2.split(img)
            tmp[0], tmp[1], tmp[2] = img[:, :, 0],img[:, :, 1], img[:, :, 2]
            X_query[i] = tmp
            self.idx2name_query[i] = img_name
        return X_ref, X_query, query_images

    def indices2image_name(self, query_results):
        # for test set
        return [[self.idx2name_ref[idx] for idx in query_result]
                for query_result in query_results]

    def make_triplet(self, X_train, y_train, nb_classes):
        X_train1, X_train2, X_train3 = np.copy(X_train), np.zeros(X_train.shape), np.zeros(X_train.shape)
        n_samples = len(X_train1)
        y_out = np.zeros((n_samples, 2))

        for row in xrange(n_samples):
            # positive sampling
            idx_of_the_same_car = map(lambda x:x[1], y_train[row])
            n_same_vehicle = len(idx_of_the_same_car)
            pos = idx_of_the_same_car[random.randint(0, n_same_vehicle-1)]
            X_train2[row] = X_train1[pos]
            # negative sampling
            neg = random.randint(0, n_samples-1)
            while neg in idx_of_the_same_car:
                neg = random.randint(0, n_samples-1)
            X_train3[row] = X_train1[neg]
            info1 = self.vid2vehicle_info[self.idx2vid_dev[row]]
            info3 = self.vid2vehicle_info[self.idx2vid_dev[neg]]
            diff_color = 1.0 if info1[0] != info3[0] else 0.0
            diff_model = 1.0 if info1[1] != info3[1] else 0.0
            # 1.0 is the basic penalty for 2 different cars.
            # 0.5 and 0.8 are hyper-parameters of color & model difference penalty.  Can be tuned.
            y_out[row, :] = 0.8 * diff_color + 0.5 * diff_model + 1.0
        X1_vid = [self.idx2vid_dev[idx] for idx in xrange(n_samples)]
        unique_ids = np.unique(X1_vid)
        original_id_2_continuous_id = {}
        for i in xrange(len(unique_ids)):
            original_id_2_continuous_id[unique_ids[i]] = i
        X1_vid = np.array([original_id_2_continuous_id[vid] for vid in X1_vid])
        X1_vid = np_utils.to_categorical(X1_vid, nb_classes)
        return (X_train1, X_train2, X_train3), (y_out, X1_vid)
