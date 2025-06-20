import torch
import os
import s3fs
from tqdm import tqdm

from huggingface_hub import HfFileSystem

MAX_LEN = 2048


def get_r2_fs():
    if os.getenv("R2_ACCESS_KEY") is None:
        raise ValueError("R2_ACCESS_KEY is not set")
    if os.getenv("R2_SECRET_KEY") is None:
        raise ValueError("R2_SECRET_KEY is not set")
    if os.getenv("R2_ENDPOINT_URL") is None:
        raise ValueError("R2_ENDPOINT_URL is not set")
    
    return s3fs.S3FileSystem(
            key=os.getenv("R2_ACCESS_KEY"),
            secret=os.getenv("R2_SECRET_KEY"),
            client_kwargs={'endpoint_url': os.getenv("R2_ENDPOINT_URL")},
            s3_additional_kwargs=dict(ACL="private")
        )

def list_local_files(path, suffixes=[".ckpt"]):
    datapaths = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapaths.append(file_path)

    # Filter out files that don't end with the suffixes (ie. when there's a HuggingFace .cache folder)
    for suffix in suffixes:
        datapaths = [f_name for f_name in datapaths if f_name.endswith(suffix)]

    return datapaths

def list_r2_files(r2_fs, path, suffixes=[".ckpt"]):
    files = r2_fs.find(path)
    return [file for file in files if file.endswith(tuple(suffixes))]

def list_hf_files(repo, suffixes=[".ckpt"]):
    hf_fs = HfFileSystem()
    datapaths = []

    if repo + "/file_list.txt" in hf_fs.ls(repo, detail=False):
        with hf_fs.open(repo + "/file_list.txt") as f:
            for line in f:
                datapaths.append(line.decode('utf-8').strip())
        return datapaths

    print(
        f"Listing files in {repo}. This is expected to take ~2 min for ShareGPT (70k files)."
    )
    for path, _, files in tqdm(hf_fs.walk(repo)):
        for file in files:
            datapaths.append(path + "/" + file)

    # Filter out files that don't end with the suffixes (ie. when there's a HuggingFace .cache folder)
    for suffix in suffixes:
        datapaths = [f_name for f_name in datapaths if f_name.endswith(suffix)]

    print(f"Found {len(datapaths)} files")
    return datapaths


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # Follow EAGLE uniform noise
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class EagleLocalDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, max_len=MAX_LEN):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        return torch.load(self.datapaths[index], weights_only=False)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"Failed to load {self.datapaths[index]} with error {e}")
            raise e
        new_data = {}

        # Squeeze due to our data generation script adding a batch dimension
        hidden_state = data["hidden_state"].squeeze(0)[: self.max_len][None, :]

        input_ids = data["input_ids"][: self.max_len][None, :]
        loss_mask = data["loss_mask"][: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data

    def set_epoch(self, epoch):
        self._epoch = epoch

class Eagle3LocalDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, max_len=MAX_LEN):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        return torch.load(self.datapaths[index], weights_only=False)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"Failed to load {self.datapaths[index]} with error {e}")
            raise e
        new_data = {}

        # Squeeze due to our data generation script adding a batch dimension
        hidden_state = data["hidden_state"].squeeze(0)[: self.max_len][None, :]
        target = data["target_hidden_states"].squeeze(0)[1 : self.max_len][None, :]

        input_ids = data["input_ids"][: self.max_len][None, :]
        loss_mask = data["loss_mask"][: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data

    def set_epoch(self, epoch):
        self._epoch = epoch


class EagleHFDataset(EagleLocalDataset):
    def __init__(self, datapath, transform=None, max_len=MAX_LEN):
        super().__init__(datapath, transform, max_len)
        self.hf_fs = HfFileSystem()

    def _open_file(self, index):
        with self.hf_fs.open(self.datapaths[index]) as f:
            return torch.load(f, weights_only=False)

class Eagle3HFDataset(Eagle3LocalDataset):
    def __init__(self, datapath, transform=None, max_len=MAX_LEN):
        super().__init__(datapath, transform, max_len)
        self.hf_fs = HfFileSystem()

    def _open_file(self, index):
        with self.hf_fs.open(self.datapaths[index]) as f:
            return torch.load(f, weights_only=False)
        
class Eagle3R2Dataset(Eagle3LocalDataset):
    def __init__(self, datapath, r2_fs, transform=None, max_len=MAX_LEN):
        super().__init__(datapath, transform, max_len)
        self.r2_fs = None
        # self.r2_fs = r2_fs

    def _open_file(self, index):
        if self.r2_fs is None:
            self.r2_fs = get_r2_fs()
        with self.r2_fs.open(self.datapaths[index]) as f:
            return torch.load(f, weights_only=False)

class DataCollatorWithPadding:
    # Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/train/main.py#L178

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features):
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_length - len(item["attention_mask"]))
                for item in features
            ]
        )
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch
