import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import models, datasets, transforms

from rcnn.utils import compute_iou, cv_img_to_tensor, label_str_to_one_hot, selective_search, tensor_to_cv_img
from rcnn.constants import VOC_2007_LABELS

class RCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        vgg = models.vgg19(weights="DEFAULT")
        self.feature_extractor = vgg.features # output: [N, 512, 7, 7] (for 224x224 image)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=classes)
        ) 

        self.bbox_regressor = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=4)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)

        logits = self.classifier(x)
        bboxes = self.bbox_regressor(x)

        return logits, bboxes

class RegionsDataset(Dataset):
    def __init__(self, img, regions, object_anns, labels, transform):
        super().__init__()
            
        self.img = img
        self.regions = regions
        self.object_anns = object_anns
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.regions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        (x, y, w, h) = self.regions[idx]
        region_box = [x, y, x+w, y+h]
        
        max_iou = 0
        matched_obj = None

        for obj in self.object_anns:
            gt_box = obj["bndbox"]
            gt_box = [int(gt_box["xmin"][0]), int(gt_box["ymin"][0]), int(gt_box["xmax"][0]), int(gt_box["ymax"][0])]
            iou = compute_iou(region_box, gt_box)

            if iou > max_iou:
                max_iou = iou
                matched_obj = obj

        # IoU >= 0.5 - match with GT bbox
        # 0.3 <= IoU < 0.5 - background 
        # IoU < 0.3 - ignore

        cropped_img = self.img[y:y+h, x:x+w] 
        cropped_img = cv_img_to_tensor(cropped_img) 
        cropped_img = self.transform(cropped_img)

        if max_iou >= 0.5 and matched_obj != None:
            label = matched_obj["name"]
            bbox = matched_obj["bndbox"]
            bbox = torch.tensor([int(bbox["xmin"][0]), int(bbox["ymin"][0]), int(bbox["xmax"][0]), int(bbox["ymax"][0])])
            one_hot_label = label_str_to_one_hot(label, self.labels)

            return cropped_img, bbox, one_hot_label
        elif max_iou < 0.5 and max_iou >= 0.3:
            label = VOC_2007_LABELS[0] 
            bbox = torch.tensor([0, 0, 0, 0])
            one_hot_label = label_str_to_one_hot(label, self.labels)

            return cropped_img, bbox, one_hot_label
        else:
            bbox = torch.tensor([0, 0, 0, 0])
            return cropped_img, bbox, -1

train_dataset = datasets.VOCDetection(root="data", year="2007", image_set="train", download=True, transform=transforms.ToTensor())
test_dataset = datasets.VOCDetection(root="data", year="2007", image_set="val", download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, 1, True)
test_loader = DataLoader(test_dataset, 1, False)

model = RCNN(classes=len(VOC_2007_LABELS))

for i, (img, annotations) in enumerate(train_loader):
    cv_img = tensor_to_cv_img(img)
    regions = selective_search(cv_img, 2000)
    
    split = int(len(regions) * 0.8)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    region_train_dataset = RegionsDataset(cv_img, regions[:split], annotations["annotation"]["object"], VOC_2007_LABELS, transform)
    region_test_dataset = RegionsDataset(cv_img, regions[split:], annotations["annotation"]["object"], VOC_2007_LABELS, transform)

    region_train_loader = DataLoader(region_train_dataset, 1, shuffle=False)
    region_test_loader = DataLoader(region_test_dataset, 1000, shuffle=True)

    for j, (img, bbox, label) in enumerate(region_train_loader): 
        if isinstance(label, int) and label == -1:
            continue
        
        output = model(img)
        print(output)
        break

    break