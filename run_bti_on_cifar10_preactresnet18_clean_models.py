# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%

output_file = "./btidbf_cifar10_preactresnet18_clean_models_part4_detection_result"

base_path = "/mnt/data/hossein/Hossein_workspace/vision_trust_worthy/downloaded_data/moein/Downloads/clean_model/"

files = os.listdir(base_path)

# Define the custom order for dataset_namees
dataset_name_order = {"cifar10": 0, "gtsrb": 1, "cifar100": 2, "tiny": 3}

def sort_key(filename):
    # Check each prefix in the desired order
    for dataset_name, rank in dataset_name_order.items():
        if filename.split("_")[0] == (dataset_name):
            return rank, filename  # sort by rank, then alphabetically
    # Files without any of these dataset_namees come after, sorted alphabetically
    return len(dataset_name_order), filename

sorted_files = sorted(files, key=sort_key)

print(sorted_files)

# %%
# base_path = "/home/hossein/.cache/kagglehub/datasets/hosseinmirzaeisa/backdoor-bench0/versions/1/"
# model_name = "cifar10_preactresnet18_sig_0_1"
# filename = os.path.join(base_path, model_name, "attack_result.pt")

def get_size_by_dataset(dataset_name):
    size = None
    if dataset_name in ["cifar10", "cifar100", "gtsrb"]:
        size = 32
    elif dataset_name == "tiny":
        size = 64
    return size


for model_name in sorted_files:
    # model_name = "cifar10_preactresnet18"
    model_arch = model_name.split("_")[1]
    if model_arch != "preactresnet18":
        continue

    filename = os.path.join(base_path, model_name, "clean_model.pth")

    

    dataset = model_name.split("_")[0]
    tlabel = 5
    attack = "clean"
    device = "cuda:0"
    size = get_size_by_dataset(dataset)
    batch_size = 128
    attack_type = "all2one"

    print(f"model_name: {model_name}, model: {model_arch}, dataset: {dataset}")

    os.system(f"python pretrain.py --filename {filename} --model_name {model_name} --dataset {dataset} --tlabel {tlabel} --model {model_arch} --attack {attack} --device {device} --size {size} --batch_size {batch_size} --attack_type {attack_type}")

    mround = 20
    uround = 30
    norm_bound = 0.3

    os.system(f"python btidbf.py --output_file {output_file} --filename {filename} --model_name {model_name} --dataset {dataset} --tlabel {tlabel} --model {model_arch} --attack {attack} --device {device} --size {size} --batch_size {batch_size} --attack_type {attack_type} --mround {mround} --uround {uround} --norm_bound {norm_bound}")


