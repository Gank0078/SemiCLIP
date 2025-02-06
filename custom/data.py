import os
import json
import h5py
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from datasets import load_dataset, load_from_disk

from training.data import DataInfo
from torchrs.datasets import RSICD, UCMCaptions, SydneyCaptions
from torchrs.datasets import UCM, WHURS19, RSSCN7, AID, RESISC45
from torchvision import transforms
from .randaugment import RandAugmentMC
import pickle
# from nltk import pos_tag
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords


OOD_NOUNS = ['lot', 'zone', 'top', 'left', 'inspection', 'way', 'brand', 'hand', 'length', 'b', 'ease',
                 'communications', 'economy', 'number', 'lies', 'towards', 'type', 'places', 'shapes', 'Lots',
                 'structure', 'sizes', 'pieces', 'stands', 'end', 'built', 'thirds', 'zones', 'bottom', 'south',
                 'prints', 'air', 'take-off', 'types', 'tens', 'fact', 'kind', 'style', 'side', 'question', 'situation',
                 'y', 'tp', 'others', 'styles', 'amount', 'drop', 'holiday', 'differnet', 'size', 'u', 'pictures',
                 'use',
                 'l', 'picture', 'looks', 'rate', 'design', 's', 'uses', 'arrange', 'z', 'piece', 'palyground', 'lots',
                 'view', 'kinds', 'j', 'work', 'beautiful', 'R', 'none', 'lack', ']', 'colourful']


def is_noun(word):
    word_tag = pos_tag([word])[0][1]
    return word_tag in ['NN', 'NNS', 'NNP', 'NNPS']

class TransformTwice(object):
    def __init__(self, transform_aug):
        resize_dim = 256
        crop_dim = 224
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.transform_aug = transform_aug
        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        aug1 = self.transform_aug(x)
        # aug2 = self.transform_aug(x)
        aug2 = self.strong(x)
        return aug1, aug2

class RSICD_CLS(RSICD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_class_info(os.path.join(self.root, "txtclasses_rsicd"))

    def load_class_info(self, class_dir):
        classes = []
        path2class = {}
        for idx, fn in enumerate(sorted(os.listdir(class_dir))):
            classes.append(fn.split(".txt")[0])
            with open(os.path.join(class_dir, fn)) as f:
                for line in f.readlines():
                    path2class[line.strip()] = idx

        self.classes = classes
        self.path2class = path2class

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        filename = self.captions[idx]["filename"]
        path = os.path.join(self.root, self.image_root, filename)
        x = self.transform(Image.open(path).convert("RGB"))
        y = self.path2class[filename]
        return x, y


class Fashion200k(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]

        txt_path = [
            f'dress_{split}_detect_all.txt',
            f'jacket_{split}_detect_all.txt',
            f'pants_{split}_detect_all.txt',
            f'skirt_{split}_detect_all.txt',
            f'top_{split}_detect_all.txt',
        ]

        data = {}
        for txt in txt_path:
            # os.path.join(self.root, 'labels', txt)
            # with open(os.path.join(self.root, txt), 'r') as f:
            with open(os.path.join(self.root, 'labels', txt), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    image_path, _, sentences = line.split('\t')
                    item_id = image_path.split('/')[3]

                    if not os.path.exists(os.path.join(self.root, image_path)):
                        continue

                    if item_id in data:
                        data[item_id]['image_path'].append(image_path)
                    else:
                        data[item_id] = dict(image_path=[image_path], sentences=sentences)
        data = [dict({'id': item_id}, **data[item_id]) for item_id in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, random.choice(item["image_path"]))
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        sentences = item['sentences']
        return dict(x=x, captions=sentences)


class FashionGen(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data, self.images = self._load_annotation_db(split)

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        # h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.hdf5")
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.h5")
        h5_file = h5py.File(h5_path)

        data = {}
        for idx in range(len(h5_file['index'])):
            item_id = int(h5_file['input_productID'][idx])
            input_name = h5_file['input_name'][idx][0]
            input_desc = h5_file['input_description'][idx][0]

            if item_id in data:
                data[item_id]['image_idx'].append(idx)
            else:
                data[item_id] = dict(image_idx=[idx], input_name=input_name, input_desc=input_desc)
        data = [dict({'id': item_id}, **data[item_id]) for item_id in data]

        images = h5_file['input_image']

        return data, images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        x = self.images[random.choice(item['image_idx'])]
        x = Image.fromarray(x)
        x = self.transform(x)

        sentences = item['input_name'].decode('latin-1') + ". "
        sentences += item['input_desc'].decode('latin-1')

        return dict(x=x, captions=sentences)


class Polyvore(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

    def _load_annotation_db(self, split):
        json_path = os.path.join(self.root, f"{split}_info.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)

        data = []
        # for item in anno_json:
        #     data.append(
        #         {
        #             "image_path": item["images"],
        #             "id": item["id"],
        #             "sentences": item["title"] + "." + item["description"],
        #             # "attributes_id": item["attributes_id"],
        #         }
        #     )

        for item in anno_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "sentences": item["title"] + "." + item["description"],
                    "attributes_id": item["attributes_id"],
                }
            )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, 'images', item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        sentences = item['sentences']
        return dict(x=x, captions=sentences)


class Fashion200k_CLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        # Remove some broken links
        self.data = [item for item in self.data if os.path.exists(os.path.join(self.root, 'women', item["image_path"]))]

        self.classes = set()
        for item in self.data:
            cls = item['class_name']
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]
        json_path = os.path.join(self.root, f"{split}_info.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)

        data = []
        for item in anno_json:
            for image_path in item['images']:
            # for image_path in item['image_path']:
                class_name = image_path.split("/")[0].replace("_", " ")
                data.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name
                    }
                )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, 'women', item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        cls_name = item['class_name']
        y = self.classes.index(cls_name)
        return x, y


class Fashion200k_SUBCLS(Fashion200k_CLS):
    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]
        json_path = os.path.join(self.root, f"{split}_info.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)

        data = []
        for item in anno_json:
            for image_path in item['images']:
            # for image_path in item['image_path']:
                class_name = image_path.split("/")[1].replace("_", " ")
                data.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name
                    }
                )

        return data


class FashionGen_CLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        self.classes = set()
        for cls in self.data['input_category']:
            cls = cls[0].decode('UTF-8').lower()
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        # h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.hdf5")
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.h5")
        data = h5py.File(h5_path)

        return data

    def __len__(self):
        return len(self.data['index'])

    def __getitem__(self, idx):
        x = self.data['input_image'][idx]
        x = Image.fromarray(x)
        x = self.transform(x)

        cls_name = self.data['input_category'][idx][0].decode('UTF-8').lower()
        y = self.classes.index(cls_name)
        return x, y


class FashionGen_SUBCLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        self.classes = set()
        for cls in self.data['input_subcategory']:
            cls = cls[0].decode('UTF-8').lower()
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        # h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.hdf5")
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.h5")
        data = h5py.File(h5_path)

        return data

    def __len__(self):
        return len(self.data['index'])

    def __getitem__(self, idx):
        x = self.data['input_image'][idx]
        x = Image.fromarray(x)
        x = self.transform(x)

        cls_name = self.data['input_subcategory'][idx][0].decode('UTF-8').lower()
        y = self.classes.index(cls_name)
        return x, y


class Polyvore_CLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        self.classes = set()
        for item in self.data:
            cls = item['class_name']
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        json_path = os.path.join(self.root, f"{split}_info.json")
        # json_path = os.path.join(self.root, f"{split}_cls_info.json")
        metadata_path = os.path.join(self.root, "polyvore_item_metadata.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)
        with open(metadata_path, 'r') as f:
            meta_json = json.load(f)

        data = []
        for item in anno_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "class_name": meta_json[str(item['id'])]['semantic_category']
                }
            )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, 'images', item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        cls_name = item['class_name']
        y = self.classes.index(cls_name)
        return x, y


class SciCap(Dataset):
    MAXLEN = 77  # maximum length for caption
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = self._init_data()

    def _init_data(self):
        image_root = os.path.join(self.root, "SciCap-No-Subfig-Img", self.split)
        json_root = os.path.join(self.root, "SciCap-Caption-All", self.split)

        samples = []
        for filename in os.listdir(json_root):
            with open(os.path.join(json_root, filename)) as f:
                json_object = json.load(f)
            if json_object["contains-subfigure"]:
                continue

            path = os.path.join(image_root, str(filename).replace("json", "png"))
            caption = json_object['0-originally-extracted']
            caption = caption[:self.MAXLEN]  # cut long captions
            samples.append([path, caption])

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, caption


class TokenizedDataset(Dataset):
    def __init__(self, dataset, indexes=None, nlen=None, selidx=-1, image_key=None, text_key=None,
                 tokenizer=None, keywords=None, itemflag=0, name=None):
        self.dataset = dataset
        self.image_key = image_key
        self.text_key = text_key
        self.tokenize = tokenizer or (lambda x: x)

        self.keywords = keywords
        self.keyword_tokens = self._init_keyword_tokens()
        self.indexes = indexes
        self.nlen = nlen
        self.selidx = selidx
        self.itemflag = itemflag
        self.name = name

    def _init_keyword_tokens(self):
        if self.keywords is not None:
            BOS, EOS = 49406, 49407
            keyword_tokens = []
            for k in self.keywords:
                k = self.tokenize(k).flatten().tolist()
                k = k[k.index(BOS) + 1: k.index(EOS)]
                keyword_tokens.append(k)
            return keyword_tokens
        else:
            return None

    def _find_keyword(self, tokens, key):
        for i in range(len(tokens)):
            idx = i  # candidate
            for j in range(len(key)):
                if tokens[i+j] != key[j]:
                    idx = None
                    break

            if idx is not None:
                return idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[int(idx)]

        if self.name == "RESISC45":
            images, cls = data
            return images, idx

        if self.indexes is not None:
            # if idx.item() in self.indexes:
            #     labels = torch.zeros(self.nlen)
            #     idxs = self.indexes[idx.item()]
            #     labels[idxs] = 1.0
            # else:
            #     labels = None
            if self.itemflag > 0:
                idx = idx
            else:
                idx = idx.item()
            if idx in self.indexes:
                labels = torch.zeros(self.nlen)
                idxs = self.indexes[idx]
                labels[idxs] = 1.0
            else:
                labels = None
        else:
            labels = None

        # read data, which is dict or list
        if isinstance(data, (list, tuple)):
            images, texts = data
        else:
            assert isinstance(data, dict)
            assert self.image_key and self.text_key
            images = data[self.image_key]
            texts = data[self.text_key]

        # tokenize captions
        if isinstance(texts, list):
            # texts = str(random.choice(texts))
            if self.selidx < 0:
                texts = str(random.choice(texts))
            else:
                texts = texts[self.selidx]
        tokens = self.tokenize([str(texts)])[0]

        if self.indexes is not None:
            # return images, tokens, labels, idx, texts
            return images, tokens, idx, labels, texts
        else:
            return images, tokens, idx, texts


def split_data(d, split_ratio, seed=42, hf_data=False):
    # set random seed
    gen = torch.Generator()
    gen.manual_seed(seed)

    # split labeled and unlabeled data
    indices = torch.randperm(len(d), generator=gen)
    size = int(len(d) * split_ratio)

    if hf_data is False:
        d1 = Subset(d, indices[:size])
        d2 = Subset(d, indices[size:])
    else:
        d1 = [d[int(i)] for i in indices[:size]]
        d2 = [d[int(i)] for i in indices[size:]]

    return d1, d2


def read_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f.readlines():
            keywords.append(line.strip())
    return keywords


def create_datainfo(args, dataset, batch_size, is_train):
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    workers = args.workers if not args.train_data else 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_custom_data(args, data, preprocess_fn, is_train, **data_kwargs):
    split = "train" if is_train else "val"

    REMOTE_SENSING_CAPTIONS = ["RSICD", "UCM", "Sydney", "RS-ALL"]
    REMOTE_SENSING_ZEROSHOT = ["RSICD-CLS", "UCM-CLS", "WHU-RS19", "RSSCN7", "AID", "RESISC45"]

    FASHION_CAPTIONS = ["Fashion200k", "FashionGen", "Polyvore", "Fashion-ALL"]
    FASHION_ZEROSHOT = ["Fashion200k-CLS", "Fashion200k-SUBCLS", "FashionGen-CLS", "FashionGen-SUBCLS", "Polyvore-CLS"]

    # if args.abs_data is not None:
    #     data_dir = args.abs_data
    # else:
    #     if args.dev == 0:
    #         data_dir = "../../datasets/"
    #     else:
    #         data_dir = "../datasets/"
    data_dir = args.data_dir
    preprocess_fn = TransformTwice(preprocess_fn)
    if data in REMOTE_SENSING_CAPTIONS:
        data_dir = data_dir + "aerial/"
        if data == "RSICD":
            d = RSICD(data_dir + "RSICD", split=split, transform=preprocess_fn)
        elif data == "UCM":
            d = UCMCaptions(data_dir + "UCM_captions", split=split, transform=preprocess_fn)
        elif data == "Sydney":
            d = SydneyCaptions(data_dir + "Sydney_captions", split=split, transform=preprocess_fn)
        elif data == "RS-ALL":
            d = ConcatDataset([
                RSICD(data_dir + "RSICD", split=split, transform=preprocess_fn),
                UCMCaptions(data_dir + "UCM_captions", split=split, transform=preprocess_fn),
                SydneyCaptions(data_dir + "Sydney_captions", split=split, transform=preprocess_fn),
            ])

        d = TokenizedDataset(d, selidx=args.selidx, image_key="x", text_key="captions", **data_kwargs)

        return d

    elif data in REMOTE_SENSING_ZEROSHOT:
        data_dir = data_dir + "aerial/"
        if data == "RSICD-CLS":
            d = RSICD_CLS(data_dir + "RSICD", split=split, transform=preprocess_fn)
        elif data == "UCM-CLS":
            d = UCM(data_dir + "UCMerced_LandUse", transform=preprocess_fn)
        elif data == "WHU-RS19":
            d = WHURS19(data_dir + "WHU-RS19", transform=preprocess_fn)
        elif data == "RSSCN7":
            d = RSSCN7(data_dir + "RSSCN7", transform=preprocess_fn)
            d.classes = [c[1:] for c in d.classes]  # "aGrass" -> "Grass"
        elif data == "AID":
            d = AID(data_dir + "AID", transform=preprocess_fn)
        elif data == "RESISC45":
            d = RESISC45(data_dir + "NWPU-RESISC45", transform=preprocess_fn)

        template = [lambda c: f"an aerial photograph of {c}."]

        if data == "RESISC45":
            d = TokenizedDataset(d, selidx=args.selidx, name="RESISC45", image_key="x", text_key="captions", **data_kwargs)
            return d
        else:
            return d, d.classes, template

    elif data in FASHION_CAPTIONS:
        data_dir = data_dir + "fashion/"
        if data == "Fashion200k":
            d = Fashion200k(data_dir + "fashion200k", split=split, transform=preprocess_fn)
        elif data == "FashionGen":
            d = FashionGen(data_dir + "FashionGen", split=split, transform=preprocess_fn)
        elif data == "Polyvore":
            d = Polyvore(data_dir + "PolyvoreOutfits", split=split, transform=preprocess_fn)
        elif data == "Fashion-ALL":
            d = ConcatDataset([
                Fashion200k(data_dir + "fashion200k", split=split, transform=preprocess_fn),
                FashionGen(data_dir + "FashionGen", split=split, transform=preprocess_fn),
                Polyvore(data_dir + "PolyvoreOutfits", split=split, transform=preprocess_fn),
            ])

        d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
        return d

    elif data in FASHION_ZEROSHOT:
        data_dir = data_dir + "fashion/"
        if data == 'Fashion200k-CLS':
            d = Fashion200k_CLS(data_dir + "fashion200k", split=split, transform=preprocess_fn)
        elif data == 'Fashion200k-SUBCLS':
            d = Fashion200k_SUBCLS(data_dir + "fashion200k", split=split, transform=preprocess_fn)
        elif data == "FashionGen-CLS":
            d = FashionGen_CLS(data_dir + "FashionGen", split=split, transform=preprocess_fn)
        elif data == 'FashionGen-SUBCLS':
            d = FashionGen_SUBCLS(data_dir + "FashionGen", split=split, transform=preprocess_fn)
        if data == "Polyvore-CLS":
            d = Polyvore_CLS(data_dir + "PolyvoreOutfits", split=split, transform=preprocess_fn)

        template = [lambda c: f"a photo of a {c}."]

        return d, d.classes, template

    else:
        if data == "SciCap":
            d = SciCap(data_dir + "science/scicap_data", split=split, transform=preprocess_fn)
            d = TokenizedDataset(d, **data_kwargs)

        elif data in ["Simpsons", "Simpsons-Captions"]:
            d = load_dataset(data_dir + "simpsons-blip-captions", keep_in_memory=True)
            image_key, text_key = "image", "text"

            def transform(batch, MAXLEN=77):
                batch[image_key] = [preprocess_fn(image) for image in batch[image_key]]
                batch[text_key] = [text[:MAXLEN] for text in batch[text_key]]
                return batch
            d.set_transform(transform)

            train_ratio = 0.9  # use 90% for training data
            d_train, d_val = split_data(d["train"], train_ratio, seed=42, hf_data=True)
            d = d_train if is_train else d_val

            d = TokenizedDataset(d, image_key=image_key, text_key=text_key, itemflag=1, **data_kwargs)

        elif data == "Simpsons-Images":
            # d = ImageFolder("/data/simpsons_dataset", transform=preprocess_fn)
            d = ImageFolder(data_dir + "simpsons_dataset", transform=preprocess_fn)

        else:
            raise ValueError(f"Unknown dataset: {data}")

        return d


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    nouns = None
    if args.train_data:
        train_kwargs = {"is_train": True, "preprocess_fn": preprocess_train, "tokenizer": tokenizer}

        if args.keyword_path is not None:
            keywords = read_keywords(args.keyword_path)
            data["keyword"] = torch.cat([tokenizer(k) for k in keywords])
            train_kwargs.update({"keywords": keywords})

        if args.train_data == "RS-SHIFT":
            d_train = get_custom_data(args, "RS-ALL", **train_kwargs)
            d_train, _ = split_data(d_train, args.label_ratio, seed=args.seed)
            # d_query, _, _ = get_custom_data(args, "RESISC45", **train_kwargs)
            d_query = get_custom_data(args, "RESISC45", **train_kwargs)
        elif args.train_data == "Simpsons":
            d_train = get_custom_data(args, "Simpsons-Captions", **train_kwargs)
            d_query = get_custom_data(args, "Simpsons-Images", **train_kwargs)
        else:
            d_train = get_custom_data(args, args.train_data, **train_kwargs)
            # d_train, nouns = get_custom_data(args, args.train_data, **train_kwargs)
            d_train, d_query = split_data(d_train, args.label_ratio, seed=args.seed)

        if args.train_data == "Simpsons":
            lab_len = len(d_train)
        else:
            lab_len = len(d_train.indices)
        if args.method.startswith('semiclip'):
            if args.train_data == "Simpsons":
                dset = d_train.dataset
            else:
                dset = d_train.dataset.dataset

            nouns_list = [[] for _ in range(lab_len)]
            for i in range(lab_len):
                if args.train_data == "SciCap":
                    texts = dset[d_train.indices[i]][1]
                elif args.train_data == "Simpsons":
                    texts = dset[i]['text']
                else:
                    texts = dset[d_train.indices[i]]['captions']
                if isinstance(texts, str):
                    text = texts
                    words = nltk.word_tokenize(text)
                    tagged_words = nltk.pos_tag(words)
                    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
                    nouns_list[i].extend(nouns)
                else:
                    for j in range(len(texts)):
                        text = texts[j]
                        words = nltk.word_tokenize(text)
                        tagged_words = nltk.pos_tag(words)
                        nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
                        nouns_list[i].extend(nouns)

                nouns_list[i] = list(dict.fromkeys(nouns_list[i]))

            nouns = [item for sublist in nouns_list for item in sublist]
            nouns = list(dict.fromkeys(nouns))
            nouns = [word for word in nouns if is_noun(word)]
            nouns = [item for item in nouns if item not in OOD_NOUNS]
            nouns_count = {word: 0 for word in nouns}
            for word_list in nouns_list:
                for word in word_list:
                    if word in nouns:
                        nouns_count[word] += 1

            nouns = [word for word in nouns if nouns_count[word] >= args.cmin and len(word) > 2 and nouns_count[word] / len(nouns_count) < 0.3]
            if args.train_data == "Simpsons":
                indexes = {idx: [] for idx in range(lab_len)}
            else:
                indexes = {idx.item(): [] for idx in d_train.indices}
            for i in range(lab_len):
                if args.train_data == "Simpsons":
                    idx = i
                else:
                    idx = d_train.indices[i].item()
                for word in nouns_list[i]:
                    if word in nouns:
                        indexes[idx].append(nouns.index(word))

            if args.train_data == "Simpsons":
                d_train = TokenizedDataset(d_train.dataset, indexes=indexes, itemflag=1, nlen=len(nouns),
                                           selidx=args.selidx, image_key="image", text_key="text", tokenizer=tokenizer)
            else:
                lab_indices = d_train.indices
                d_train = TokenizedDataset(d_train.dataset.dataset, indexes=indexes, nlen=len(nouns),
                                           selidx=args.selidx, image_key="x", text_key="captions", tokenizer=tokenizer)
                d_train = Subset(d_train, lab_indices)

        data["train"] = create_datainfo(args, d_train, args.batch_size // 2, is_train=True)
        data["query"] = create_datainfo(args, d_query, (args.batch_size // 2) * args.mu, is_train=True)
        data["nouns"] = nouns

    if args.val_data:
        d_val = get_custom_data(args, args.val_data, preprocess_val, is_train=False, tokenizer=tokenizer)
        data["val"] = create_datainfo(args, d_val, args.batch_size, is_train=False)

    if args.imagenet_val is not None:
        d_zeroshot, classnames, template = get_custom_data(args, args.imagenet_val, preprocess_val, is_train=False)
        data["zeroshot-val"] = create_datainfo(args, d_zeroshot, args.batch_size, is_train=False)
        data["classnames"] = classnames
        data["template"] = template

    return data
