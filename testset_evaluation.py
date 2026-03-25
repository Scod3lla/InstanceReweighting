import torch
from models.resnet import *

from models.wide_resnet import *

import os
from models import *
import json

import argparse
from autoattack import AutoAttack

import utils.lab_const as lab_const
import utils.constants as const
import attack_generator as attack


import torchvision
import socket
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TransformSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        return self.transform(im), labels

    def __len__(self):
        return len(self.indices)


def load_model(backbone, num_classes):
    if backbone == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif backbone == "resnet50":
        return ResNet50(num_classes=num_classes)
    elif backbone == "resnet152":
        return ResNet152(num_classes=num_classes)
    elif backbone == "wideresnet34":
        return Wide_ResNet(num_classes=num_classes)


# Function used for loading the base torch data
def _load_torch_data(dataset_name, train=True):

    # Loading the dataset folder based on the local machine
    if socket.gethostname() in lab_const.hostnames:
        DATASET_FOLDER = lab_const.dataset_folder_dict[socket.gethostname()]

    else:
        exit()
    
    # Checking in the dataset is supported and, eventually, loading the dataset
    if dataset_name in const.supported_datasets:
        if dataset_name == 'cifar10':
            subset = torchvision.datasets.CIFAR10(root=DATASET_FOLDER, train=train, download=True)
        elif dataset_name == "mnist":
            subset = torchvision.datasets.MNIST(root=DATASET_FOLDER, train=train, download=True)
        elif dataset_name == "svhn":
            split = 'train' if train else 'test'
            subset = torchvision.datasets.SVHN(root=DATASET_FOLDER, split=split, download=True)
    else:
        # TODO: Customize the Exception
        raise Exception(f"Dataset {dataset_name} not supported. Use one of the following: {const.supported_datasets}")
    
    return subset

def load_test_data(dataset_name, batch_size, use_cuda, test_set_size=None):

    # Choosing the kwargs based on the cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # Loading the torchvision dataset
    testset = _load_torch_data(dataset_name, train=False)

    # Slicing the test set
    indices = torch.arange(len(testset))
    test_indices = indices[:test_set_size]

    # Create the testing
    test_subset = TransformSubset(testset, test_indices, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, **kwargs)

    # Returning testing and validation sets
    return test_loader


# -------------------------------------------------------------------------
def load_weights(exp_folder, backbone, dataset, device, epoch=None):
    
    network = load_model(backbone=backbone, num_classes=const.num_classes_dict[dataset])

    # Checkpoint path
    if epoch is None or epoch == 'best':
        checkpoint_path = os.path.join(exp_folder, f'checkpoint_best.pt')
    elif epoch == 'last':
        checkpoint_path = os.path.join(exp_folder, f'checkpoint_last.pt')
    else:
        checkpoint_path = os.path.join(exp_folder, f'checkpoint_epoch{epoch}.pt')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Update weights
    network.load_state_dict(checkpoint['net'])

    return network


def get_autoattack_member(member, net, epsilon, device, seed):
    adversary = AutoAttack(net, norm='Linf', eps=epsilon, version='custom', device=device, seed=seed)
    adversary.attacks_to_run = [member]
    return adversary

# -------------------------------------------------------------------------


def eval_standard_autoattack(exp_folder, backbone, dataset, epsilon, epoch=None,
                               cuda=None, seed=0, test_set_size=None, batch_size=256, full_mode=False):

    # Setting up the CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cuda}" if use_cuda else "cpu")

    # Loading the weights
    net = load_weights(exp_folder, backbone, dataset, device, epoch)
    net = net.to(device)
    net.eval()

    adversary = AutoAttack(net, norm='Linf', eps=epsilon, version='standard')


    # Loading the test data
    test_set_size = 3000 if full_mode else None
    test_loader = load_test_data(dataset_name=dataset, test_set_size=test_set_size, batch_size=batch_size, use_cuda=use_cuda)

    # Instantiating the attack result tensors
    attack_memeber_results = torch.zeros(len(test_loader.dataset), dtype=torch.long)-1
    whole_targets = torch.zeros(len(test_loader.dataset), dtype=torch.long)-2
    
    # Iterating over the batches
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        current_batch_size = targets.shape[0]

        x_test, y_test = inputs.to(device), targets.to(device)
        adv_x = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

        adv_outputs = net(adv_x)
        _, adv_y = adv_outputs.max(1)
                    # Saving the results

        attack_memeber_results[(batch_idx*batch_size):(batch_idx*batch_size)+current_batch_size] = adv_y
        whole_targets[(batch_idx*batch_size):(batch_idx*batch_size)+current_batch_size] = targets


    attack_results_accuracies= {"autoattack": ((attack_memeber_results == whole_targets).sum() / len(whole_targets)).item()}

    with open(os.path.join(exp_folder, f"{'full_' if full_mode else ''}autoattack.json"), 'w') as f:
        json.dump(attack_results_accuracies, f, indent=2)




def eval_memberwise_autoattack(exp_folder, backbone, dataset, epsilon, epoch=None,
                               cuda=None, seed=0, test_set_size=None, batch_size=256, full_mode=False):

    # Setting up the CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cuda}" if use_cuda else "cpu")

    print(device)

    loss_type = "trades" if "trades" in exp_folder else "at"

    # Loading the weights
    net = load_weights(exp_folder, backbone, dataset, device, epoch)
    net = net.to(device)
    net.eval()

    # Loading the test data
    test_set_size = None if full_mode else 3000
    test_loader = load_test_data(dataset_name=dataset, test_set_size=test_set_size, batch_size=batch_size, use_cuda=use_cuda)
    
    # Instantiating the individual attack members
    adversaries = {
        'clean': None,
        "pgd":  attack.GA_PGD,
        'apgd-ce': get_autoattack_member('apgd-ce', net, epsilon, device, seed),
        'apgd-t': get_autoattack_member('apgd-t', net, epsilon, device, seed),
        'square': get_autoattack_member('square', net, epsilon, device, seed),
        'fab-t': get_autoattack_member('fab-t', net, epsilon, device, seed)
    }

    # attack_results = {attack: [] for attack in range(4)}
    attack_results = {}

    attack_results_accuracies = {}

    autoattack_results = torch.ones(len(test_loader.dataset), dtype=torch.long)


    for atk_name, adversary in adversaries.items():

        # Instantiating the attack result tensors
        attack_memeber_results = torch.zeros(len(test_loader.dataset), dtype=torch.long)-1
        whole_targets = torch.zeros(len(test_loader.dataset), dtype=torch.long)-2
        
        # Iterating over the batches
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print(f"Batch iter [{batch_idx}/{len(test_loader)}]")
            current_batch_size = targets.shape[0]
            x_test, y_test = inputs.to(device), targets.to(device)
            if adversary is None:
                print("++++ CLEAN ACCURACY ++++")
                logits = net(x_test)
                probabilities = (logits).softmax(dim=1)
                adv_y = probabilities.argmax(dim=1)

            elif atk_name == "pgd":
                print("++++ PGD 100 iterations ++++")

                adv_x, _ = adversary(net,x_test,y_test,0.031, 0.007, 100,
                                     loss_fn="cent",
                                     category="Madry",
                                     rand_init=True)
                
                adv_outputs = net(adv_x)
                _, adv_y = adv_outputs.max(1)

            else:
                _, adv_y = adversary.run_standard_evaluation(x_test, y_test, return_labels=True, bs=batch_size)
            
            # Saving the results
            attack_memeber_results[(batch_idx*batch_size):(batch_idx*batch_size)+current_batch_size] = adv_y
            whole_targets[(batch_idx*batch_size):(batch_idx*batch_size)+current_batch_size] = targets

        # Adding the current attack results to the whole result dictionary
        attack_results[atk_name] = attack_memeber_results
        attack_results_accuracies[atk_name] = ((attack_memeber_results == whole_targets).sum() / len(whole_targets)).item()

        # Calcolo anche la rob accuracy di autoattack
        if atk_name != "pgd":
            autoattack_results &= attack_memeber_results == whole_targets

    
    # Adding the targets to the whole result dictionary
    attack_results['target'] = whole_targets
    attack_results["autoattack"] = autoattack_results

    attack_results_accuracies["autoattack"] =  ((autoattack_results).sum() / len(whole_targets)).item()
    
    # Saving the results
    model_evaluated = epoch
    if full_mode:
        model_evaluated = 'full_' + model_evaluated
    else:
        model_evaluated = 'light_' + model_evaluated

    attack_results_path = os.path.join(exp_folder, f"{model_evaluated}_autoattack_memberwise_results.pt")
    torch.save(attack_results, attack_results_path)
    

    with open(os.path.join(exp_folder, f"{model_evaluated}_robust_accuracies.json"), 'w') as f:
        json.dump(attack_results_accuracies, f, indent=2)


# -------------------------------------------------------------------------


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        dest="base_dir",
        type=str,
        default=("")
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "last"]
    )

    parser.add_argument("--full_mode", action="store_true", help="Usare la modalità full (flag)")

    args = parser.parse_args()
    base_dir = args.base_dir


    paths = []
    files = os.listdir(base_dir)

    epoch = args.checkpoint


    for root, dirs, files in os.walk(base_dir):
        if "checkpoints" in dirs:
            checkpoints_dir = os.path.join(root, "checkpoints")

            checkpoint_files = os.listdir(checkpoints_dir)
     
            checkpoint_name = f"checkpoint_{args.checkpoint}.pt"

            if checkpoint_name not in checkpoint_files:
                print(f"{checkpoint_files}")
                print(f"In [{checkpoints_dir}] NO '{checkpoint_name}' found")
                continue


            result_prefix = f"{'full' if args.full_mode else 'light'}_{args.checkpoint}"
            result_file = f"{result_prefix}_autoattack_memberwise_results.pt"

            if result_file not in checkpoint_files:
                print(f"Evaluation missing: {args.full_mode=} of '{checkpoints_dir}'")
                paths.append(checkpoints_dir)
                  

    for dir in paths:
        sub_list = dir.split("/")
        dataset = sub_list[2]
        backbone = sub_list[1]


        print(dataset, backbone)
        print(dir)
        
        eval_memberwise_autoattack(
                                    exp_folder=dir,
                                    epoch=epoch,
                                    backbone=backbone,
                                    dataset=dataset,
                                    epsilon=0.0314,
                                    cuda=0,
                                    batch_size=512,
                                    full_mode=args.full_mode
                                    )