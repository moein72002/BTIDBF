import  os

from PIL import Image
import csv

import torch.utils.data as data

class GTSRB(data.Dataset):
    def __init__(self, data_root, train, transform = None):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(data_root, "Train")
            self.images, self.labels = self._get_data_train_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        else:
            self.data_folder = os.path.join(data_root, "Test")
            self.images, self.labels = self._get_data_test_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)

        self.transform = transform

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + '' + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_handle = Image.open(self.images[index])
        image = im_handle.copy()
        im_handle.close()
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label