import torch
from datasets.GTSRB import GTSRB
from datasets.Tiny_Imagenet import TinyImageNet
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from models import densenet, resnet, vgg
from models import vit
from models.ia.models import Generator
import cifar


class Box():
    def __init__(self, opt) -> None:
        self.opt = opt
        self.filename = opt.filename
        self.dataset = opt.dataset
        self.num_classes = self.get_num_classes(self.dataset)
        self.tlabel = opt.tlabel
        self.model = opt.model
        self.attack = opt.attack
        self.normalizer = self.get_normalizer()
        self.denormalizer = self.get_denormalizer()
        self.size = opt.size
        self.device = opt.device
        # self.num_classes = opt.num_classes
        self.attack_type = opt.attack_type
        self.root = opt.root
        if self.attack_type == "all2all":
            self.res_path = self.dataset + "-" + self.attack + "-" + self.model + "-targetall"
        elif self.attack_type == "all2one":
            self.res_path = self.dataset + "-" + self.attack + "-" + self.model + "-target" + str(self.tlabel)
    
    def get_save_path(self):
        save_path = os.path.join(self.root, "results/"+self.res_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return save_path

    def get_num_classes(self, dataset_name):
        if dataset_name in ["mnist", "cifar10"]:
            num_classes = 10
        elif dataset_name == "gtsrb":
            num_classes = 43
        elif dataset_name == "celeba":
            num_classes = 8
        elif dataset_name == 'cifar100':
            num_classes = 100
        elif dataset_name == 'tiny':
            num_classes = 200
        elif dataset_name == 'imagenet':
            num_classes = 1000
        else:
            raise Exception("Invalid Dataset")
        return num_classes

    def get_normalizer(self):
        dataset_name = self.dataset
        # idea : given name, return the default normalization of images in the dataset
        if dataset_name == "cifar10":
            # from wanet
            dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
        elif dataset_name == 'cifar100':
            '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
            dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]))
        elif dataset_name == "mnist":
            dataset_normalization = (transforms.Normalize([0.5], [0.5]))
        elif dataset_name == 'tiny':
            dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
        elif dataset_name == "gtsrb" or dataset_name == "celeba":
            dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
        elif dataset_name == 'imagenet':
            dataset_normalization = (
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        else:
            raise Exception("Invalid Dataset")
        return dataset_normalization

    def get_denormalizer(self):
        # Get the normalizer which holds the original mean and std values
        normalizer = self.get_normalizer()
        mean = normalizer.mean
        std = normalizer.std

        # Compute the inverse transformation parameters:
        # For each channel, denorm_mean = -mean / std and denorm_std = 1 / std.
        denorm_mean = [-m / s for m, s in zip(mean, std)]
        denorm_std = [1 / s for s in std]

        return transforms.Normalize(denorm_mean, denorm_std)
        
    def get_transform(self, train=True, random_crop_padding=4):
        dataset_name = self.dataset
        input_height, input_width = self.size, self.size
        # idea : given name, return the final implememnt transforms for the dataset
        transforms_list = []
        transforms_list.append(transforms.Resize((input_height, input_width)))
        if train:
            transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=random_crop_padding))
            # transforms_list.append(transforms.RandomRotation(10))
            if dataset_name == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip())

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(self.get_normalizer(dataset_name))
        return transforms.Compose(transforms_list)

    def get_dataloader(self, train, batch_size, shuffle):
        tf = self.get_transform(train)
        dataset_name = self.dataset
        if dataset_name == "cifar10":
            # Load the CIFAR-10 test dataset (downloads to "./data" if not already present)
            ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
        elif dataset_name == "cifar100":
            # Load the CIFAR-10 test dataset (downloads to "./data" if not already present)
            ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf)
        elif dataset_name == "tiny":
            ds = TinyImageNet("/mnt/data/hossein/Hossein_workspace/vision_trust_worthy/downloaded_data/moein/Downloads/tiny_imagenet_dataset",
                                                            split='val',
                                                            download=True, transform=tf
                                                            )
        elif dataset_name == "gtsrb":
            ds = GTSRB("/mnt/data/hossein/Hossein_workspace/vision_trust_worthy/downloaded_data/moein/Downloads/gtsrb/gtsrb",
                                                    train=False, transform=tf
                                                    )

        # if self.dataset == "cifar10":
        #     if train == "clean":
        #         ds = cifar.CIFAR(path=os.path.join(self.root, "datasets/cifar10"), train=True, train_type=0, tf=tf)
        #     elif train == "poison":
        #         ds = cifar.CIFAR(path=os.path.join(self.root, "datasets/cifar10"), train=True, train_type=1, tf=tf)
        #     else:
        #         ds = cifar.CIFAR(path=os.path.join(self.root, "datasets/cifar10"), train=False, tf=tf)

        # elif self.dataset == "imagenet":
        #     ds = imagenet.ImageNet(path=os.path.join(self.root, "datasets"), train=train, tf=tf)
        
        # elif self.dataset == "gtsrb":
        #     if train == "clean":
        #         ds = gtsrb.GTSRB(path=os.path.join(self.root, "datasets/gtsrb"), train=True, train_type=0, tf=tf)
        #     elif train == "poison":
        #         ds = gtsrb.GTSRB(path=os.path.join(self.root, "datasets/gtsrb"), train=True, train_type=1, tf=tf)
        #     else:
        #         ds = gtsrb.GTSRB(path=os.path.join(self.root, "datasets/gtsrb"), train=False, tf=tf)

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=6)
        return dl

    def get_state_dict(self):
        filename = self.filename

        state_dict = torch.load(filename, map_location=torch.device('cpu'))

        classifier = self.get_model()
        classifier.load_state_dict(state_dict["model"])
        
        classifier = classifier.to(self.device)
        classifier.eval()

        param1 = None
        param2 = None

        return param1, param2, classifier
    
    def get_model(self):

        if self.model == "densenet":
            return densenet.DenseNet121(num_classes=self.num_classes)
        
        elif self.model == "resnet18":
            return resnet.ResNet18(num_classes=self.num_classes)
            
        elif self.model == "vgg16":
            return vgg.VGG("VGG16", num_classes=self.num_classes)
        
        elif self.model == "vit":
            return vit.ViT(image_size = self.size,
                           patch_size = 4,
                           num_classes = self.num_classes,
                           dim = int(512),
                           depth = 6,
                           heads = 8,
                           mlp_dim = 512,
                           dropout = 0.1,
                           emb_dropout = 0.1)

    
