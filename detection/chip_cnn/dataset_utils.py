import os
import torch
import pandas as pd
import cv2
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import PIL
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib as mpl



class ChipsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, which_colors=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        if which_colors is None:
            self.which_colors = ['green', 'black', 'red', 'blue']
        else:
            self.which_colors = which_colors
        self.images_metadata_list = self.get_img_paths(root_dir)

    def get_img_paths(self, root_dir):
        images_metadata = []
        for d in os.listdir(root_dir):
            d_path = os.path.join(root_dir, d)
            if os.path.isdir(d_path):
                d_split = d.split('-')
                if len(d_split) == 2:
                    color = d_split[0]
                    num_chips = int(d_split[1])
                    if color in self.which_colors and num_chips in [j + 1 for j in range(5)]:
                        for filepath in [os.path.join(d_path, f) for f in os.listdir(d_path) if f[:7] == 'img_num']:
                            images_metadata.append((color, num_chips, filepath))

        return images_metadata

    def __len__(self):
        return len(self.images_metadata_list)

    def __getitem__(self, idx):
        color, num_chips, img_path = self.images_metadata_list[idx]
        image = cv2.imread(img_path)
        # image_gray = image[:, :, 0]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'num_chips': torch.tensor(num_chips - 1, dtype=torch.int64)}
        return sample


class Net(nn.Module):
    def __init__(self, keep_prob=0.2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 12 * 16, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 5)

    def forward(self, x, keep_prob=None):
        x = self.pool(F.relu(self.conv1(x)))
        # if keep_prob is not None:
        #     x = torch.nn.functional.dropout(x, p=keep_prob, training=True)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 12 * 16)
        x = F.relu(self.fc1(x))
        if keep_prob is not None:
            x = torch.nn.functional.dropout(x, p=keep_prob, training=True)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


######################################################################
# Let's instantiate this class and iterate through the data samples. We
# will print the sizes of first 4 samples and show their landmarks.
#
def show_dataset(dataset, n=3, n_examples=2):
    img = np.vstack((np.hstack((np.asarray(dataset[i]['image']) for _ in range(n)))
                     for i in range(n_examples)))

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return

chips_dataset = ChipsDataset('./data', transform=torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(mode=None),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.RandomAffine(15, translate=(0.001, 0.01), scale=(0.95, 1.05), resample=PIL.Image.BILINEAR,
                                        fillcolor=0),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
# chips_dataset = ChipsDataset('./data', transform=None)
# print(len(chips_dataset))
#
# img = chips_dataset[1]['image']
# label = chips_dataset[1]['num_chips']
# # print(img.dtype, img.shape, label)
# plt.tight_layout()
# # plt.imshow(img)
# # plt.show()

# show_dataset(chips_dataset)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(mode=None),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.RandomAffine(15, translate=(0.001, 0.01), scale=(0.95, 1.05), resample=PIL.Image.BILINEAR,
                                        fillcolor=0),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(mode=None),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = ChipsDataset('./data', transform=train_transform, which_colors=['blue', 'black', 'red'])
testset = ChipsDataset('./data', transform=test_transform, which_colors=['green'])
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=2)
classes = [i + 1 for i in range(5)]

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('optimization routine constructed')

batches_count = 0
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data['image'], data['num_chips']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, keep_prob=0.1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        batches_count += 1
        if batches_count % 100 == 99:    # print every 31 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        if batches_count % 200 == 199:    # print every 31 mini-batches
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data['image'], data['num_chips']
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

print('Finished Training')


######################################################################
# However, we are losing a lot of features by using a simple ``for`` loop to
# iterate over the data. In particular, we are missing out on:
#
# -  Batching the data
# -  Shuffling the data
# -  Load the data in parallel using ``multiprocessing`` workers.
#
# ``torch.utils.data.DataLoader`` is an iterator which provides all these
# features. Parameters used below should be clear. One parameter of
# interest is ``collate_fn``. You can specify how exactly the samples need
# to be batched using ``collate_fn``. However, default collate should work
# fine for most use cases.
#

# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)


# Helper function to show a batch
# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch, landmarks_batch = \
#             sample_batched['image'], sample_batched['landmarks']
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)
#
#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
#
#     for i in range(batch_size):
#         plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
#                     landmarks_batch[i, :, 1].numpy(),
#                     s=10, marker='.', c='r')
#
#         plt.title('Batch from dataloader')
#
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['landmarks'].size())
#
#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break

######################################################################
# Afterword: torchvision
# ----------------------
#
# In this tutorial, we have seen how to write and use datasets, transforms
# and dataloader. ``torchvision`` package provides some common datasets and
# transforms. You might not even have to write custom classes. One of the
# more generic datasets available in torchvision is ``ImageFolder``.
# It assumes that images are organized in the following way: ::
#
#     root/ants/xxx.png
#     root/ants/xxy.jpeg
#     root/ants/xxz.png
#     .
#     .
#     .
#     root/bees/123.jpg
#     root/bees/nsdf3.png
#     root/bees/asd932_.png
#
# where 'ants', 'bees' etc. are class labels. Similarly generic transforms
# which operate on ``PIL.Image`` like  ``RandomHorizontalFlip``, ``Scale``,
# are also available. You can use these to write a dataloader like this: ::
#
#   import torch
#   from torchvision import transforms, datasets
#
#   data_transform = transforms.Compose([
#           transforms.RandomSizedCrop(224),
#           transforms.RandomHorizontalFlip(),
#           transforms.ToTensor(),
#           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#       ])
#   hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                              transform=data_transform)
#   dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                                batch_size=4, shuffle=True,
#                                                num_workers=4)
#
# For an example with training code, please see
# :doc:`transfer_learning_tutorial`.

