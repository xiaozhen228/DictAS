import torch.utils.data as data
import json
from PIL import Image
import numpy as np
import torch
import os
import albumentations as A
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import cv2
import glob
import torchvision.transforms as transforms


class Makedataset():
	def __init__(self, train_data_path, preprocess_test, mode, image_size = 518):
		self.train_data_path= train_data_path
		self.preprocess_test = preprocess_test
		self.mode = mode
		self.target_transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.CenterCrop(image_size),transforms.ToTensor()])

	def make_dataset(self, name, product_list, batchsize, args, k_shot = 1, shuf = True):
		dataset = MyDataset(root=self.train_data_path, transform=self.preprocess_test, target_transform=self.target_transform,
											mode =self.mode , product_list= product_list, dataset = name, k_shot = k_shot, args = args)
		obj_list = dataset.get_cls_names()
		
		dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = shuf)
		
		return dataloader, obj_list

class MyDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='train_self', k_shot=0, dataset = None, args = None, product_list = None):
		if mode == "val":
			k_shot = 1
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.mode = mode
		self.use_unified_few = True
		self.k_shot = k_shot
		self.random_choose = False
		self.dataset = dataset
		self.args = args


		anomaly_source_path = self.args.anomaly_source_path
		self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))
		self.resize_shape = (512,512)

		img_trans_best = A.Compose([
			A.RandomRotate90(p = 1),
			A.Rotate(limit=[30, 270], p=1.0),
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5),
			A.GridDropout(ratio=0.3, p=0.5),
			A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
		], is_check_shapes=False)

		self.img_trans_best = img_trans_best
		
		if os.path.exists(f"./fix_few_path/{dataset}/fix_{k_shot}-shot.txt"):
			self.few_data_path = f"./fix_few_path/{dataset}/fix_{k_shot}-shot.txt"
		else:
			self.few_data_path = None

		self.data_all = []

		if  mode == "train_self":
			meta_info = json.load(open(f'{self.root}/meta_{self.dataset}.json', 'r'))
			self.cls_names = list(meta_info["train"].keys())
			#self.cls_names = ["chewinggum"]
			for cls_name in self.cls_names:
				self.data_all.extend(meta_info["train"][cls_name])

	
		
		elif mode == "val":
			meta_info = json.load(open(f'{self.root}/meta_{self.dataset}.json', 'r'))
			keys = meta_info["test"].keys()
			keys = list(keys)
			if product_list is not None:
				for product in product_list:
					assert product in keys
				for key in keys:
					if key not in product_list:
						del meta_info["test"][key]
			self.cls_names = list(meta_info["test"].keys())
			#self.cls_names = ["bottle"]
			for cls_name in self.cls_names:
				self.data_all.extend(meta_info["test"][cls_name])
			
		else:
			meta_info = json.load(open(f'{self.root}/meta_{self.dataset}.json', 'r'))
			self.cls_names = list(meta_info["test"].keys())
			#self.cls_names = ["screw"]
			for cls_name in self.cls_names:
				self.data_all.extend(meta_info["test"][cls_name])
			
		self.length = len(self.data_all)

		if not self.random_choose:
			if self.use_unified_few and (self.mode == "test" or self.mode == "val"):
				if self.few_data_path is not None:
					self.few_data_list = self.load_fixed_normal()
				else:
					self.few_data_list = self.choose_fixed_normal()
		else:
			self.few_data_list = self.choose_fixed_normal()

		self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

		self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
	def augment_image(self, image, mask, anomaly_source_path):
		
		image = np.array(image, dtype= np.uint8)[:, :, ::-1]
		mask = np.array(mask, dtype= np.uint8)
		image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[1]), interpolation=cv2.INTER_CUBIC)
		image = image / 255.0
		mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[1]),interpolation=cv2.INTER_NEAREST )
		h, w = image.shape[0], image.shape[1]

		aug = self.randAugmenter()
		perlin_scale = 3
		min_perlin_scale = 1
		anomaly_source_img = cv2.imread(anomaly_source_path)
		anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(w, h))

		anomaly_img_augmented = aug(image=anomaly_source_img)
		perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
		perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

		perlin_noise = rand_perlin_2d_np((h, w), (perlin_scalex, perlin_scaley))
		perlin_noise = self.rot(image=perlin_noise)
		threshold = 0.5
		perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
		perlin_thr = np.expand_dims(perlin_thr, axis=2)

		img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

		beta = torch.rand(1).numpy()[0] * 0.8

		augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
			perlin_thr)

		no_anomaly = torch.rand(1).numpy()[0]
		if no_anomaly > 1:
			image = np.array(image*255, dtype= np.uint8)[:, :, ::-1]
			msk = np.array(mask, dtype= np.uint8)* 255
			image = Image.fromarray(image)
			msk = Image.fromarray(msk, mode = "L")
			return image, msk , False
		else:
			augmented_image = augmented_image.astype(np.float32)
			msk = (perlin_thr).astype(np.float32)
			augmented_image = msk * augmented_image + (1-msk)*image

			msk = np.squeeze(msk) + mask
			msk[msk!=0] = 1.0
			has_anomaly = True
			if np.sum(msk) == 0:
				has_anomaly= False
			
			augmented_image = np.array(augmented_image*255, dtype= np.uint8)[:, :, ::-1]
			msk = np.array(msk, dtype= np.uint8) * 255
			augmented_image = Image.fromarray(augmented_image)
			msk = Image.fromarray(msk, mode = "L")
			#return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
			return augmented_image, msk, has_anomaly

	def randAugmenter(self):
		aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
		aug = iaa.Sequential([self.augmenters[aug_ind[0]],
								self.augmenters[aug_ind[1]],
								self.augmenters[aug_ind[2]]]
								)
		return aug
	
	def load_fixed_normal(self):
		few_dict = {}
		for index, cls_name in enumerate(self.cls_names):
			few_dict[cls_name] = []
			f = open(self.few_data_path, "r")
			for line in f:
				if cls_name in line and cls_name != "fryum":
					few_dict[cls_name].append(line.rstrip())
					assert os.path.exists(os.path.join(self.root,line.rstrip()))
				if cls_name == "fryum" and "fryum" in line and "pipe_fryum" not in line:
					few_dict[cls_name].append(line.rstrip())
					assert os.path.exists(os.path.join(self.root,line.rstrip()))
			assert len(few_dict[cls_name]) == self.k_shot, cls_name
			f.close()
		return few_dict

	def choose_fixed_normal(self):
		few_dict = {}
		save_dir = os.path.join(self.args.save_path, f"{self.args.k_shot}_{self.args.seed}.txt")
		meta_info = json.load(open(f'{self.root}/meta_{self.dataset}.json', 'r'))
		meta_info = meta_info["train"]
		f = open(save_dir, "w")
		for index, cls_name in enumerate(self.cls_names):
			few_dict[cls_name] = []
			data_temp = meta_info[cls_name]
			indices = np.random.choice(len(data_temp), size= self.k_shot, replace=False)
			for i in range(len(indices)):
				few_dict[cls_name].append(data_temp[indices[i]]["img_path"])
				f.write(data_temp[indices[i]]["img_path"] + '\n')
		f.close()
		return few_dict


	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def Trans_good(self, img , img_mask):
		img_mask = np.array(img_mask)
		img = np.array(img)[:, :, ::-1]
		#augmentations = self.img_trans_good(mask=img_mask, image=img)
		augmentations = self.img_trans_best(mask=img_mask, image=img)
		img = augmentations["image"][:, :, ::-1]
		img_mask = augmentations["mask"]
		img = Image.fromarray(img)
		img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
		return img, img_mask
	

	def rotate_image(self, image, angle):
		
		transform = A.Rotate(limit=(angle, angle), 
							always_apply=True)
		transformed = transform(image=image)
		return transformed['image']

	def Roate_support(self, img_list, angle_range=(-180, 180), step=45):
		img_list_new = []
		
		for i in range(len(img_list)):
			img_list_new.append(img_list[i])
			img = np.array(img_list[i])[:,:,::-1]
			for angle in range(angle_range[0], angle_range[1] + 1, step):
				rotated_img = self.rotate_image(img, angle)[:,:,::-1]
				img_list_new.append(Image.fromarray(rotated_img))
		return img_list_new
	


	def __getitem__(self, index):
		data = self.data_all[index]
		if self.mode == "train_self":
			img_ano_path, mask_ano_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
																data['specie_name'], data['anomaly']
			img_ano = Image.open(os.path.join(self.root, img_ano_path)).convert("RGB")
			if anomaly == 1: 
				img_ano_mask = np.array(Image.open(os.path.join(self.root, mask_ano_path)).convert('L')) > 0
				img_ano_mask = Image.fromarray(img_ano_mask.astype(np.uint8) * 255, mode='L')
			else:
				img_ano_mask = Image.fromarray(np.zeros((img_ano.size[1], img_ano.size[0])), mode='L')

		
			img_ano = img_ano.resize((1024, 1024), Image.BICUBIC)
			img_ano_mask = img_ano_mask.resize((1024, 1024), Image.NEAREST)

			
			is_trans = torch.rand(1).numpy()[0] 
			if is_trans > 0.1:    # 0.2
				img_good, img_good_mask = self.Trans_good(img_ano, img_ano_mask)
			else:
				img_good, img_good_mask  = img_ano, img_ano_mask


			is_gen = torch.rand(1).numpy()[0]

			if is_gen > self.args.gen_anomaly_rate:   # 0.7
				anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
				img_ano, img_ano_mask, anomaly = self.augment_image(img_ano, img_ano_mask, self.anomaly_source_paths[anomaly_source_idx])
			img_ano = self.transform(img_ano) if self.transform is not None else img_ano 
			img_ano_mask = self.target_transform(
				img_ano_mask) if self.target_transform is not None and img_ano_mask is not None else img_ano_mask
			
			img_good = self.transform(img_good) if self.transform is not None else img_good

			img_good_mask = self.target_transform(
				img_good_mask) if self.target_transform is not None and img_good_mask is not None else img_good_mask
			
			return {'img_ano': img_ano, 'img_ano_mask': img_ano_mask, 'img_good': img_good, 'img_good_mask':img_good_mask, 'cls_name': cls_name,  "anomaly":anomaly}

		elif self.mode == "test" or self.mode == "val":
			img_ano_path, mask_ano_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
													data['specie_name'], data['anomaly']
			img_good_path_list = self.few_data_list[cls_name]
			img_ano = Image.open(os.path.join(self.root, img_ano_path)).convert("RGB")
			if anomaly == 1: 
				img_ano_mask = np.array(Image.open(os.path.join(self.root, mask_ano_path)).convert('L')) > 0
				img_ano_mask = Image.fromarray(img_ano_mask.astype(np.uint8) * 255, mode='L')
			else:
				img_ano_mask = Image.fromarray(np.zeros((img_ano.size[1], img_ano.size[0])), mode='L')

			img_good_list = [Image.open(os.path.join(self.root, img_good_path)).convert("RGB") for img_good_path in img_good_path_list]
			
			if cls_name == "screw" and self.k_shot <=4:
				img_good_list = self.Roate_support(img_good_list)
			img_good_mask_list = [Image.fromarray(np.zeros((img_good.size[1], img_good.size[0])), mode='L') for img_good in img_good_list]

			img_ano = self.transform(img_ano) if self.transform is not None else img_ano 
			img_ano_mask = self.target_transform(
				img_ano_mask) if self.target_transform is not None and img_ano_mask is not None else img_ano_mask
			img_good_list = [self.transform(img_good) for img_good in img_good_list]
			img_good_mask_list = [self.target_transform(img_good_mask) for img_good_mask in img_good_mask_list]
			img_good = torch.stack(img_good_list, dim = 0)
			img_good_mask= torch.stack(img_good_mask_list, dim = 0)
			return {'img_ano': img_ano, 'img_ano_mask': img_ano_mask, 'img_good': img_good, 'img_good_mask':img_good_mask, 'cls_name': cls_name, "img_ano_path": os.path.join(self.root, img_ano_path), "anomaly":anomaly, "ano_path":img_ano_path }
		
		elif self.mode == "test_zero":
			img_ano_path, mask_ano_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
										data['specie_name'], data['anomaly']
			img_ano = Image.open(os.path.join(self.root, img_ano_path)).convert("RGB")
			if anomaly == 1: 
				img_ano_mask = np.array(Image.open(os.path.join(self.root, mask_ano_path)).convert('L')) > 0
				img_ano_mask = Image.fromarray(img_ano_mask.astype(np.uint8) * 255, mode='L')
			else:
				img_ano_mask = Image.fromarray(np.zeros((img_ano.size[1], img_ano.size[0])), mode='L')
			img_good, img_good_mask = self.Trans_good(img_ano, img_ano_mask)
			img_good_list = [img_good]
			img_good_mask_list = [img_good_mask]
			img_ano = self.transform(img_ano) if self.transform is not None else img_ano 
			img_ano_mask = self.target_transform(
				img_ano_mask) if self.target_transform is not None and img_ano_mask is not None else img_ano_mask
			img_good_list = [self.transform(img_good) for img_good in img_good_list]

			img_good_mask_list = [self.target_transform(img_good_mask) for img_good_mask in img_good_mask_list]
			img_good = torch.stack(img_good_list, dim = 0)
			img_good_mask= torch.stack(img_good_mask_list, dim = 0)

			return {'img_ano': img_ano, 'img_ano_mask': img_ano_mask, 'img_good': img_good, 'img_good_mask':img_good_mask, 'cls_name': cls_name, "anomaly":anomaly, "img_ano_path": os.path.join(self.root, img_ano_path)}

