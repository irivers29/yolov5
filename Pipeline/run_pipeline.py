import pickle
import sys
from pickletools import uint8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.autograd import Variable

sys.path.append('/home/ignacio/code/MasterThesis/Instance Recognition/yolov5')
import detect_ros as detect

sys.path.append('/home/ignacio/code/MasterThesis/Instance Recognition/Siamese')
from mydataset import DatasetSKU110K
from SiameseModel import SiameseNetwork
from torch.utils.data import DataLoader

image = cv2.imread('/home/ignacio/code/MasterThesis/Instance Recognition/yolov5/test/now/shelf4.png')
target = cv2.imread('/home/ignacio/code/MasterThesis/Instance Recognition/Data/Targets/Coke/shelf0.jpg')


def _process_inference_image(img, image_size = 416):
    resized_img = cv2.resize(img, dsize=(image_size, image_size),
                                interpolation=cv2.INTER_CUBIC)

    
    RGB_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2GRAY)
    numpy_array = np.asarray(RGB_image)

    #numpy_array = np.moveaxis(numpy_array, -1, 0)
    numpy_array = torchvision.transforms.functional.to_tensor(numpy_array)
    return numpy_array

def calc_prob(image1, image2, model):
    image1 = _process_inference_image(image1, image_size=64)
    image2 = _process_inference_image(image2, image_size=64)

    val1, val2 = Variable(image1), Variable(image2)
    
    val1 = val1[None,:]
    val2 = val2[None,:]

    output_val = model.forward(val1, val2)

    
    return output_val


#plt.imshow(image)
#plt.show()
#print(type(image))

path = '/home/ignacio/code/MasterThesis/Instance Recognition/Data/Results/results.picl'

objectRep = open(path, "rb")
weights = pickle.load(objectRep)

net = SiameseNetwork()
net.load_state_dict(weights["state_dict"])

#import yolov5.detect as detect
crops = detect.run_from_python(source=image)
for i in crops:
    print(i)
    coordinates = crops[i][0]
    cropped_img = crops[i][1]

    x_low = int(coordinates[0].item())
    y_low = int(coordinates[1].item())
    x_high = int(coordinates[2].item())
    y_high = int(coordinates[3].item())

    window_name = 'Image'

    new_image = image[y_low:y_high, x_low:x_high]
    cv2.imshow(window_name, new_image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    output_val = calc_prob(new_image, target, net)
    print(output_val)
  
    start_point = (x_low, y_low)
    end_point = (x_high, y_high)
    color = (255, 0, 0)
    thickness = 2
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, start_point, end_point, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    # Displaying the image 
    #cv2.imshow(window_name, masked) 
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()

#trainSet = DatasetSKU110K(samples_per_class = 1, image_size = 64, transform = None, inference = True, image1 = new_image, image2 = target)
#InferenceLoader = DataLoader(trainSet, batch_size = 1)









