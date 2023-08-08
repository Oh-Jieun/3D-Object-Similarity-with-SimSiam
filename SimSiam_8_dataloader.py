import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SimSiamDataloader(Dataset):
    def __init__(self, dataset_dir):
        self.root_dir = dataset_dir
        self.image_names = []
        self.labels = []
        self.class_to_idx = {}
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

        idx = 0
        for class_folder in os.listdir(dataset_dir):
            self.class_to_idx[class_folder] = idx
            class_folder_path = os.path.join(dataset_dir, class_folder)

            for obj_folder in os.listdir(class_folder_path):
                obj_folder_path = os.path.join(class_folder_path, obj_folder)

                img_list = []
                imgs = [file for file in os.listdir(obj_folder_path) if file.endswith(".png")]

                if len(imgs) != 8:
                    print(f"Warning: Folder '{obj_folder}' has not 8 images.")
                    continue
                
                for img in imgs:
                    image_path = os.path.join(obj_folder_path, img)
                    img_list.append(image_path)
                self.image_names.append(img_list)
                self.labels.append(idx)
            idx += 1
    
    def img_read(self, path_):
        try:
            img = Image.open(path_).convert("RGB")
        except:
            img = Image.fromarray(np.unit8(np.random.rand(224, 224, 3) * 255))
        return self.transforms(img)


    def __getitem__(self, idx):
        image_names = self.image_names[idx]

        img1 = self.img_read(image_names[0])
        img2 = self.img_read(image_names[1])
        img3 = self.img_read(image_names[2])
        img4 = self.img_read(image_names[3])
        img5 = self.img_read(image_names[4])
        img6 = self.img_read(image_names[5])
        img7 = self.img_read(image_names[6])
        img8 = self.img_read(image_names[7])

        labels = self.labels[idx]


        return img1, img2, img3, img4, img5, img6, img7, img8, labels

    def __len__(self):
        return len(self.image_names)
        