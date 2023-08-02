import json
import copy
import numpy as np
from pycocotools import mask
import cv2
import os
import sys
from PIL import Image
from skimage import measure
import shutil
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from datetime import datetime
import torch
new_train_jsonPath = "./mo/annotations/new_train.json"
new_test_jsonPath = "./mo/annotations/new_test.json"
new_val_jsonPath = "./mo/annotations/new_val_10p.json"
json_path="./mo/annotations/new.json"
json_10p_s_path="./mo/annotations/instances_train2017_10p_s.json"
json_10p_w_path="./mo/annotations/instances_train2017_10p_w.json"
json_50p_s_path="./mo/annotations/instances_train2017_50p_s.json"
json_50p_w_path="./mo/annotations/instances_train2017_50p_w.json"
train_img_path="./cell_data_root/coco/train2017"
test_img_path="./cell_data_root/coco/val2017"
val_10p_img_path="./cell_data_root/coco/val_10p"
img_path="./mo/JPEGImages"


# 生成mask
def maskToanno(ground_truth_mask,ann_count,segmentation_id):
  unique_values = np.unique(ground_truth_mask)  # 获取图像中的唯一非零值
  unique_values = unique_values[unique_values != 0] 
  annotations = []  # 一幅图片所有的annotations
  for value in unique_values:
    x = np.zeros((1000, 1000))
    mask = (ground_truth_mask == value).astype(np.uint8)  # 创建新的二值mask，仅包含当前value对应的部分
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 根据二值mask找到轮廓
    annotation = {
        "segmentation": [contours[0].flatten().tolist()],
        "area": cv2.contourArea(contours[0]),
        "iscrowd": 0,
        "image_id": ann_count,
        "bbox": list(cv2.boundingRect(contours[0])),
        "category_id": 1,
        "id": segmentation_id
    }
    annotations.append(annotation)
    segmentation_id += 1
  return annotations,segmentation_id
def get_annotations(input_filename, output_filename,block_mask_path):
  block_mask_image_files = os.listdir(block_mask_path)
  with open(input_filename, "r") as f:
    coco_json = json.load(f)
  coco_points_json = copy.deepcopy(coco_json)
  coco_annos = coco_points_json.get("annotations")
  coco_annos=[]
  annCount = 1
  segmentation_id=1
  for mask_img in block_mask_image_files:
    block_im=np.asarray(Image.open(os.path.join(block_mask_path,mask_img)))
    block_anno,segmentation_id=maskToanno(block_im,annCount,segmentation_id)
    coco_annos.extend(block_anno)
    annCount=annCount+1
  coco_points_json["annotations"]=coco_annos
  with open(output_filename, "w") as f:
    json.dump(coco_points_json, f)

# 生成点
def get_point_annotations(dataset_dir, seed:int,num_points_per_instance=2):
  '''
  dataset:"./data_root/coco/annotations",
  num_points_per_instance:每个实例点的数量
  '''
  # 设置随机种子
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)
  s = "new_train"
  print(
      "Start sampling {} points per instance for annotations {}.".format(
          num_points_per_instance, s
      )
  )
  input_filename=os.path.join(dataset_dir, "{}.json".format(s))
  output_filename=os.path.join(
              dataset_dir,
              "{}_n{}_without_masks.json".format(s, num_points_per_instance),
          )
  with open(input_filename, "r") as f:
    coco_json = json.load(f)
  coco_annos = coco_json.pop("annotations")
  coco_points_json = copy.deepcopy(coco_json)

  imgs = {}
  for img in coco_json["images"]:
    imgs[img["id"]] = img

  new_annos = []
  for ann in coco_annos:
    # convert mask
    t = imgs[ann["image_id"]]
    h, w = t["height"], t["width"]
    segm = ann.get("segmentation")
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    mask = mask_utils.decode(rle)
    new_ann = copy.deepcopy(ann)
    # sample points in image coordinates
    box = ann["bbox"]
    point_coords_wrt_image = np.random.rand(num_points_per_instance, 2)
    point_coords_wrt_image[:, 0] = point_coords_wrt_image[:, 0] * box[2]
    point_coords_wrt_image[:, 1] = point_coords_wrt_image[:, 1] * box[3]
    point_coords_wrt_image[:, 0] += box[0]
    point_coords_wrt_image[:, 1] += box[1]
    # round to integer coordinates
    point_coords_wrt_image = np.floor(point_coords_wrt_image).astype(int)
    # get labels
    # assert (point_coords_wrt_image >= 0).all(), (point_coords_wrt_image, mask.shape)
    assert (point_coords_wrt_image[:, 0] < w).all(), (point_coords_wrt_image, mask.shape)
    assert (point_coords_wrt_image[:, 1] < h).all(), (point_coords_wrt_image, mask.shape)
    point_labels = mask[point_coords_wrt_image[:, 1], point_coords_wrt_image[:, 0]]
    # store new annotations
    new_ann["point_coords"] = point_coords_wrt_image.tolist()
    new_ann["point_labels"] = point_labels.tolist()
    new_annos.append(new_ann)
  coco_points_json["annotations"] = new_annos

  with open(output_filename, "w") as f:
    json.dump(coco_points_json, f)

  print("{} is modified and stored in {}.".format(input_filename, output_filename))

# 将图片img属性的补全
def format_filename(n:int,json_path:str):
  with open(json_path, 'r') as file:
    data = json.load(file)
  data['images']=[]
  images=data['images']
  for i in range(n):
      my_dict = {}
      my_dict['file_name'] = str(i+1).zfill(6) + ".jpg"
      my_dict['height'] = 1000
      my_dict['width'] = 1000
      my_dict['id'] = i+1
      images.append(my_dict)
  # 写回JSON文件
  with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)

# 加入信息
def add_info(json_path:str):
  with open(json_path, 'r') as file:
    data = json.load(file)
  data["info"]={"description": "COCO 2017 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2017,"contributor": "COCO Consortium","date_created": "2017/09/01"}
  data["licenses"]=[{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},{"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},{"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},{"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},{"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},{"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}]
    # 写回JSON文件
  with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)

# 分割测试与训练
def divide_train_test(json_path,new_train_jsonPath,new_test_jsonPath,split_ratio=0.9):
  # 读取原始JSON文件
    with open(json_path, 'r') as file:
        data = json.load(file)


    # 随机打乱图像列表
    images = data['images']
    random.shuffle(images)
    # 计算分割点
    split_index = int(len(images) * split_ratio)
    
    # 分割训练集和测试集
    train_data = copy.deepcopy(data)
    train_data['images'] = images[:split_index]
    train_data['images'].sort(key=lambda x: x['id'])
    test_data = copy.deepcopy(data)
    test_data['images'] = images[split_index:]
    test_data['images'].sort(key=lambda x: x['id'])
    # 删除不属于训练集的annotations项
    train_image_ids = set(image['id'] for image in train_data['images'] )
    train_data['annotations'] = [annotation for annotation in train_data['annotations'] if annotation['image_id'] in train_image_ids]

    # 删除不属于测试集的annotations项
    test_image_ids = set(image['id'] for image in test_data['images'])
    test_data['annotations'] = [annotation for annotation in test_data['annotations'] if annotation['image_id'] in test_image_ids]

    # 写回训练集JSON文件
    with open(new_train_jsonPath, 'w') as file:
        json.dump(train_data, file, indent=4)

    # 写回测试集JSON文件
    with open(new_test_jsonPath, 'w') as file:
        json.dump(test_data, file, indent=4)

# 分离子集
def divide_subsets(json_path,json_path_s,json_path_w,split_ratio:int):
    # 读取原始JSON文件
    with open(json_path, 'r') as file:
        data = json.load(file)
      # 随机打乱图像列表
    images = data['images']
    random.shuffle(images)
        # 计算分割点
    split_index = int(len(images) * split_ratio)
    json_s=copy.deepcopy(data)
    json_w=copy.deepcopy(data)
    json_s['images'] = images[:split_index]
    json_s['images'].sort(key=lambda x: x['id'])
        # 删除不属于验证集的annotations项
    json_s_image_ids = set(image['id'] for image in json_s['images'] )
    json_s['annotations'] = [annotation for annotation in json_s['annotations'] if annotation['image_id'] in json_s_image_ids]
    json_w['images'] = images[split_index:]
    json_w['images'].sort(key=lambda x: x['id'])
        # 删除不属于验证集的annotations项
    json_w_image_ids = set(image['id'] for image in json_w['images'] )
    json_w['annotations'] = [annotation for annotation in json_w['annotations'] if annotation['image_id'] in json_w_image_ids]   
    # 写回强标签JSON文件
    with open(json_path_s, 'w') as file:
        json.dump(json_s, file, indent=4)

    # 写回弱标签JSON文件
    with open(json_path_w, 'w') as file:
        json.dump(json_w, file, indent=4)
    
# 分离图片
def divide_img(img_path,train_img_path,json_path):
  # 判断train_img_path路径下是否存在.jpg文件
  if os.path.exists(train_img_path):
      files = os.listdir(train_img_path)
      jpg_files = [file for file in files if file.endswith('.jpg')]
      
      # 删除.train_img_path路径下的所有.jpg文件
      for file in jpg_files:
          file_path = os.path.join(train_img_path, file)
          os.remove(file_path)
  with open(json_path, 'r') as file:
    data = json.load(file)
  imagefiles=data["images"]
  imagenames=[i["file_name"] for i in imagefiles]
  # 遍历img_path路径下的文件
  for filename in os.listdir(img_path):
    file_path = os.path.join(img_path, filename)
    # 检查当前文件名是否出现在imagenames中
    if filename in imagenames:
      # 复制文件到train_img_path路径下
      shutil.copy(file_path, train_img_path)    

# 从训练集分离验证集
def divide_val_from_train(new_train_jsonPath,new_val_jsonPath,val_img_path,train_img_path,split_ratio=0.5):
  # 读取原始JSON文件
    with open(new_train_jsonPath, 'r') as file:
        data = json.load(file)
  # 随机打乱图像列表
    images = data['images']
    random.shuffle(images)
    # 计算分割点
    split_index = int(len(images) * split_ratio)
    val_data = copy.deepcopy(data)
    val_data['images'] = images[:split_index]
    val_data['images'].sort(key=lambda x: x['id'])
    # 删除不属于验证集的annotations项
    val_image_ids = set(image['id'] for image in val_data['images'] )
    val_data['annotations'] = [annotation for annotation in val_data['annotations'] if annotation['image_id'] in val_image_ids]
    imagefiles=val_data["images"]
    # 写回训练集JSON文件
    with open(new_val_jsonPath, 'w') as file:
        json.dump(val_data, file, indent=4)
    if os.path.exists(val_img_path):
      files=os.listdir(val_img_path)
      jpg_files=[file for file in files if file.endswith('.jpg')]
      for file in jpg_files:
        file_path=os.path.join(val_img_path,file)
        os.remove(file_path)
    imagenames=[i["file_name"] for i in imagefiles]
    # 遍历img_path路径下的文件
    for filename in os.listdir(train_img_path):
      file_path = os.path.join(train_img_path, filename)
      # 检查当前文件名是否出现在imagenames中
      if filename in imagenames:
        # 复制文件到train_img_path路径下
        shutil.copy(file_path, val_img_path)  
        
# 分割可视化
def sem_view(json_path,img_dir):
  coco = COCO(json_path)
  # 获取所有图像ID
  image_ids = coco.getImgIds()
  # 可视化每张图像及其对应的分割标注
  for image_id in image_ids:
    # 获取图像文件名
    img_info = coco.loadImgs(image_id)[0]
    img_file = img_dir + '/' + img_info['file_name']
    
    # 读取图像
    image = plt.imread(img_file)
    
    # 获取图像对应的分割标注
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    
    # 在图像上绘制分割结果
    masks=[]
    for ann in annotations:
        # 获取分割掩码
        mask = coco.annToMask(ann)
        masks.append(mask)
        # 绘制分割结果
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='gray')
        plt.axis('off')
        plt.show()

# 点框可视化
def cell_point_view(json_file,dataset_dir,view_point:bool):
  with open(json_file, 'r') as f:
    coco_data = json.load(f)
  images_info = coco_data['images']
  # 获取所有的标注信息
  annotations_info = coco_data['annotations']
  # 循环遍历每个图像
  for image_info in images_info:
    # 拼接图像路径
    img_path = dataset_dir + image_info['file_name']
    # 找到当前图像的标注信息
    image_id = image_info['id']
    annotations = [anno for anno in annotations_info if anno['image_id'] == image_id]
    # 读取图像
    image = np.array(Image.open(img_path))
    # 创建画布
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    # 遍历每个标注
    for annotation in annotations:
        bbox = annotation['bbox']
        # 绘制边界框
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        if view_point:
          point_coords = annotation['point_coords']
          point_labels = annotation['point_labels']
          for coord, label in zip(point_coords, point_labels):
              x, y = coord[0], coord[1]
              # 根据点标签设置不同颜色和形状的点
              if label == 1:
                  color, marker,size = 'red', 'o',2
              elif label == 0:
                  color, marker,size= 'blue', 'x',2
              else:
                  color, marker,size = 'green', '+',2
              plt.scatter(x, y, c=color, marker=marker,s=size)  
    # 显示图像
    plt.show()
    
