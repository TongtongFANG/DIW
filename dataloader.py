import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import errno
import codecs
import numpy as np
import torch
import torchvision.transforms as transforms
from utils import noisify


# Dataset for Fashion-MNIST
class F_MNIST(data.Dataset):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'f_raw'
    processed_folder = 'f_processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_type=None,
                 noise_rate=0.2, random_state=100):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'f_mnist'
        self.noise_type = noise_type

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

            if noise_type != 'clean':
                self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(train_labels=self.train_labels,
                                                                          noise_type=noise_type, noise_rate=noise_rate,
                                                                          random_state=random_state)
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                _train_labels = [i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):

        if self.train:
            # if self.noise_type is not None:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed.copy()).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed.copy()).view(length, num_rows, num_cols)


# Dataloader for Fashion-MNIST
class F_MNIST_dataloader():
    def __init__(self, batch_size, batch_size_val, noise_type, noise_rate, num_val):
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_val = num_val

    def run(self):
        train_dataset = F_MNIST(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=self.noise_type,
                                noise_rate=self.noise_rate
                                )

        test_dataset = F_MNIST(root='./data/',
                               download=True,
                               train=False,
                               transform=transforms.ToTensor(),
                               noise_type=self.noise_type,
                               noise_rate=self.noise_rate
                               )

        val_dataset = F_MNIST(root='./data/',
                              download=True,
                              train=True,
                              transform=transforms.ToTensor(),
                              noise_type='clean',
                              noise_rate=0
                              )

        # random select val data
        val_idx = random.sample(range(len(train_dataset)), self.num_val)
        # remove val data from train data
        data_train = [i for j, i in enumerate(train_dataset) if j not in val_idx]

        data_val = []
        for i in val_idx:
            # prepare val dataset
            data_val.append(val_dataset[i])
            # add val data back to train data
            label = val_dataset[i][1].item()
            list_val = [val_dataset[i][0], label, val_dataset[i][2]]
            data_train.append(list_val)

        train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                                   batch_size=self.batch_size,
                                                   num_workers=0,
                                                   drop_last=True,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  num_workers=0,
                                                  drop_last=True,
                                                  shuffle=False)

        val_loader = torch.utils.data.DataLoader(dataset=data_val,
                                                 batch_size=self.batch_size_val,
                                                 num_workers=0,
                                                 drop_last=True,
                                                 shuffle=True)

        return train_loader, val_loader, test_loader
