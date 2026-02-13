# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "VLCS_gen",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class MyDataset(Dataset):
    def __init__(self, root_real, root_gen, transform) -> None:
        super().__init__()
        self.root_real = root_real 
        self.root_gen = root_gen

        self.dataset_real = ImageFolder(self.root_real, transform)
        self.dataset_gen = []
        for rg in self.root_gen:
            self.dataset_gen.append(ImageFolder(rg, transform))

    def __len__(self):
        return len(self.dataset_real)

    def __getitem__(self, index):
        res = {}
        sample_r, target_r = self.dataset_real[index]
        res["x"] = sample_r
        res["y"] = target_r
        for i in range(len(self.dataset_gen)):
            sample_g, target_g = self.dataset_gen[i][index]
            res["x_g%d" % i] = sample_g
            res["y_g%d" % i] = target_g
        # sample_g, target_g = self.dataset_gen[index]
        return res

class MyDataset2(Dataset):
    def __init__(self, root_real, root_gen, transform, N=5, text_dir='', text_dir2='') -> None:
        super().__init__()
        self.root_real = root_real 
        self.root_gen = root_gen
        self.N = N

        self.dataset_real = ImageFolder(self.root_real, transform)

        classes = [d.name for d in os.scandir(text_dir) if d.is_dir()]
        classes.sort()
        self.classes = classes
        self.all_text = []
        self.all_text2 = []
        for cls in classes:
            target_dir = os.path.join(text_dir, cls)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    with open(path, 'r') as f:
                        prompt = f.readline()    
                    # self.all_text.append(prompt.split(' ')[4])
                    self.all_text.append(prompt)

            for cls in classes:
                target_dir = os.path.join(text_dir2, cls)
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        with open(path, 'r') as f:
                            prompt = f.readline()    
                        # self.all_text.append(prompt.split(' ')[4])
                        self.all_text2.append(prompt)

        # self.dataset_gen = []
        # for rg in self.root_gen:
        #     self.dataset_gen.append(ImageFolder(rg, transform))
        self.style_prompts =  [
            "abstract style",
            "realistic style",
            "impressionistic style",
            "cubist style",
            "expressionist style",
            "futuristic style",
            "pop art style",
            "cartoon style",
            "photorealism style",
            "fantasy style",
            "sci-fi style",
            "sketch style",
            "art style",
        ]

        self.background_prompts =  [
            'in the mountains',
            'on the beach',
            'in the desert'
            'in the cityscape',
            'in the fields',
            'in the jungle',
            'in the countryside',
            'under the sky',
            'under the water',
            'at dawn',
            'at sunrise',
            'at daylight',
            'at nightfall',
            'at moonlight',
            'at brightness',
            'at darkness',
        ]

    def __len__(self):
        return len(self.dataset_real)

    def __getitem__(self, index):
        res = {}
        sample_r, target_r = self.dataset_real[index]
        res["x"] = sample_r
        res["y"] = target_r
        res["text"] = []
        for i in range(self.N):
            style_word = random.choice(self.style_prompts)
            background_word = random.choice(self.background_prompts)
            if self.all_text2:    
                res["text"].append(f"{self.all_text[index]}, {self.all_text2[index]}, {style_word}, {background_word}")
            else:
                res["text"].append(f"{self.all_text[index]}, {style_word}, {background_word}")
            # res["text"].append(f"a {style_word} image of a {self.all_text[index]} {background_word}")
            # res["text"].append(f"{self.all_text[index]}, {style_word}, {background_word}")
            # res["text"].append(f"{style_word}, {background_word}")
        # sample_g, target_g = self.dataset_gen[index]
        return res

class MyDataset3(Dataset):
    """
    Real Image + Gen Image + Gen prompt
    """
    def __init__(self, root_real, root_gen, root_prompt, transform) -> None:
        super().__init__()
        self.root_real = root_real 
        self.root_gen = root_gen
        self.root_prompt = root_prompt

        self.dataset_real = ImageFolder(self.root_real, transform)
        self.dataset_gen = []
        for rg in self.root_gen:
            self.dataset_gen.append(ImageFolder(rg, transform))

        classes = [d.name for d in os.scandir(root_real) if d.is_dir()]
        classes.sort()
        self.all_prompt = []

        for k, text_dir in enumerate(self.root_prompt):
            self.all_prompt.append([])
            for cls in classes:
                target_dir = os.path.join(text_dir, cls)
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        with open(path, 'r') as f:
                            prompt = f.readline() 
                        self.all_prompt[k].append(prompt)

    def __len__(self):
        return len(self.dataset_real)
    
    def __getitem__(self, index):
        res = {}
        sample_r, target_r = self.dataset_real[index]
        res["x"] = sample_r
        res["y"] = target_r
        for i in range(len(self.dataset_gen)):
            sample_g, _ = self.dataset_gen[i][index]
            prompt_g = self.all_prompt[i][index]
            res["x_g%d" % i] = sample_g
            res["p_g%d" % i] = prompt_g
        # sample_g, target_g = self.dataset_gen[index]
        return res

class MultipleDomainDataset:
    N_STEPS = 5000          # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override
    def __getitem__(self, index):
        return self.datasets[index]
    def __len__(self):
        return len(self.datasets)

class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]: #domains list
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            ) 
class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']
class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']
class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams, root_gen=None, train_gen=None, text_dir='', text_dir2='', prompt_dir=None, clip_transform=None):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments) # list of all domains in the dataset, in sorted order

        if clip_transform is None:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # transforms.Normalize(
                #     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.Normalize(
                #     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            transform = clip_transform
            augment_transform = clip_transform

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            text_path = os.path.join(text_dir, environment)
            if train_gen is None or i in test_envs:
                if root_gen is None:
                    env_dataset = ImageFolder(path,
                        transform=env_transform)
                else: 
                    path_gen = []
                    for rg in root_gen:
                        path_gen.append(os.path.join(rg, environment))
                    path_prompt = []
                    if prompt_dir is not None:
                        for rp in prompt_dir:
                            path_prompt.append(os.path.join(rp, environment))
                    # env_dataset = MyDataset2(path, path_gen,
                    #     transform=env_transform, text_dir=text_path, text_dir2=text_dir2)
                    env_dataset = MyDataset3(path, path_gen, path_prompt,
                        transform=env_transform)
            else:
                path_gen = []
                for rg in train_gen:
                    path_gen.append(os.path.join(rg, environment))
                path_prompt = []
                if prompt_dir is not None:
                    for rp in prompt_dir:
                        path_prompt.append(os.path.join(rp, environment))
                # env_dataset = MyDataset2(path, path_gen,
                #     transform=env_transform, text_dir=text_dir, text_dir2=text_dir2)
                env_dataset = MyDataset3(path, path_gen, path_prompt,
                        transform=env_transform)
            ################################ Code required for RCERM ################################ 
            # env_dataset: <class 'torchvision.datasets.folder.ImageFolder'>, 
            # Dataset ImageFolder
            #     Number of datapoints: 2050
            #     Root location: ../../DG/DomainBed/domainbed/data/PACS/art_painting
            # Dataset ImageFolder
            #     Number of datapoints: 2345
            #     Root location: ../../DG/DomainBed/domainbed/data/PACS/cartoon
            # Dataset ImageFolder
            #     Number of datapoints: 1671
            #     Root location: ../../DG/DomainBed/domainbed/data/PACS/photo
            # Dataset ImageFolder
            #     Number of datapoints: 3934
            #     Root location: ../../DG/DomainBed/domainbed/data/PACS/sketch
            # access it via:
            ################################ Code required for RCERM ################################ 
            self.datasets.append(env_dataset)
            ################################ Code required for RCERM ################################ 
#             # env_dataset containst first all elts of class 0, then class 1, ...class C-1
#             for (batch_idx, sample_batched) in enumerate(self.datasets[i]):
#             ###     if batch_idx==1:
#             ###         break
#                 print('im ',batch_idx,' :',sample_batched)
#                 ## eg, im  0  : (<PIL.Image.Image image mode=RGB size=227x227 at 0x7FB44448C430>, 0)
            ################################ Code required for RCERM ################################ 

            

        self.input_shape = (3, 224, 224,)
        if root_gen is None:
            self.num_classes = len(self.datasets[-1].classes)
        else:
            self.num_classes = len(self.datasets[-1].dataset_real.classes)

class VLCS(MultipleEnvironmentImageFolder):
    # for ViT
    CHECKPOINT_FREQ = 1000
    N_STEPS = 30000
    # for ATViT
    # CHECKPOINT_FREQ = 700
    # N_STEPS = 11643

    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams, root_gen=None, text_dir='', text_dir2='', prompt_dir=None):
        self.dir = os.path.join(root, "VLCS/")
        if root_gen is not None: 
            root_gens = []
            for i in range(5):
                # root_gens.append(os.path.join(root_gen, "PACS_cn_gen%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "VLCS_t2i_s%d/" % (i+1)))
                root_gens.append(os.path.join(root_gen, "VLCS_kd%d/" % (i+1)))
            root_gen = root_gens
            # root_gen = os.path.join(root_gen, "PACS/")
        if prompt_dir is not None:
            root_prompt = []
            for i in range(5):
                root_prompt.append(os.path.join(prompt_dir, "VLCS_txt_kd%d" % (i+1)))
            prompt_dir = root_prompt
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, root_gen, text_dir=text_dir, text_dir2=text_dir2, prompt_dir=prompt_dir)

class VLCS_gen(MultipleEnvironmentImageFolder):
    # for ViT
    # CHECKPOINT_FREQ = 100
    # N_STEPS = 1769
    # for ATViT
    CHECKPOINT_FREQ = 250
    # N_STEPS = 11643
    N_STEPS = 2500

    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams, root_gen=None, train_gen=None, text_dir='', text_dir2='', prompt_dir=None, clip_transform=None):
        self.dir = os.path.join(root, "VLCS/")
        if root_gen is not None: 
            root_gens = []
            for i in range(5):
                # root_gens.append(os.path.join(root_gen, "PACS_cn_gen%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "VLCS_t2i_s%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "VLCS_clip%d/" % (i+1)))
                root_gens.append(os.path.join(root_gen, "VLCS_kd%d/" % (i+1)))
            root_gen = root_gens
        # if train_gen is not None: 
        #     train_gens = []
        #     for i in range(5):
        #         # root_gens.append(os.path.join(root_gen, "PACS_cn_gen%d/" % (i+1)))
        #         train_gens.append(os.path.join(train_gen, "VLCS_train_gen%d/" % (i+1)))
        #     train_gen = train_gens
        if prompt_dir is not None:
            root_prompt = []
            for i in range(5):
                root_prompt.append(os.path.join(prompt_dir, "VLCS_txt_kd%d" % (i+1)))
            prompt_dir = root_prompt
        train_gen = None
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, root_gen, train_gen, text_dir=text_dir, text_dir2=text_dir2, clip_transform=clip_transform)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    # N_STEPS = 8350
    # N_STEPS = 9932
    N_STEPS = 30000
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams, root_gen=None, train_gen=None, text_dir='', text_dir2='', prompt_dir=None, clip_transform=None):
        self.dir = os.path.join(root, "PACS/")
        if root_gen is not None: 
            root_gens = []
            for i in range(5):
                # root_gens.append(os.path.join(root_gen, "PACS_cn_gen%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "PACS_clip%d/" % (i+1)))
                root_gens.append(os.path.join(root_gen, "PACS_kd%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "PACS_wotp_clip%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "PACS_abd1_gen%d/" % (i+1)))
            root_gen = root_gens
            # if train_gen is not None: 
            #     train_gens = []
            #     for i in range(5):
            #         # root_gens.append(os.path.join(root_gen, "PACS_cn_gen%d/" % (i+1)))
            #         train_gens.append(os.path.join(train_gen, "PACS_train_gen%d/" % (i+1)))
            #     train_gen = train_gens
            train_gen = None
            if prompt_dir is not None:
                root_prompt = []
                for i in range(5):
                    root_prompt.append(os.path.join(prompt_dir, "PACS_txt_kd%d" % (i+1)))
                prompt_dir = root_prompt
            # root_gen = os.path.join(root_gen, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, root_gen, train_gen, text_dir=text_dir, text_dir2=text_dir2, prompt_dir=prompt_dir, clip_transform=clip_transform)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    # CHECKPOINT_FREQ = 1000
    # N_STEPS = 16454
    CHECKPOINT_FREQ = 30000
    N_STEPS = 30000
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams, root_gen=None, train_gen=None, text_dir='', text_dir2='', prompt_dir=None, clip_transform=None):
        self.dir = os.path.join(root, "office_home/")
        if root_gen is not None: 
            root_gens = []
            for i in range(5):
                # root_gens.append(os.path.join(root_gen, "office_home_cn_gen%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "office_home_clip%d/" % (i+1)))
                root_gens.append(os.path.join(root_gen, "office_home_kd%d/" % (i+1)))
                # root_gens.append(os.path.join(root_gen, "office_home_wotp_clip%d/" % (i+1)))
            root_gen = root_gens
        # if train_gen is not None: 
        #     train_gens = []
        #     for i in range(5):
        #         # root_gens.append(os.path.join(root_gen, "PACS_cn_gen%d/" % (i+1)))
        #         train_gens.append(os.path.join(train_gen, "office_home_train_gen%d/" % (i+1)))
        #     train_gen = train_gens
        train_gen = None
        if prompt_dir is not None:
            root_prompt = []
            for i in range(5):
                root_prompt.append(os.path.join(prompt_dir, "office_home_txt_kd%d" % (i+1)))
            prompt_dir = root_prompt
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, root_gen, train_gen, text_dir=text_dir, text_dir2=text_dir2, prompt_dir=prompt_dir, clip_transform=clip_transform)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

