# from torch.utils.data import Dataset
# import torch

# class eyes_dataset(Dataset):
#     def __init__(self, x_file_path, y_file_path, transform=None):
#         self.x_files = x_file_path
#         self.y_files = y_file_path
#         self.transform = transform
        
#     def __getitem__(self, idx):
#         x = self.x_files[idx]
#         x = torch.from_numpy(x).float()
        
#         y = self.y_files[idx]
#         y = torch.from_numpy(y).float()
        
#         return x,y
    
#     def __len__(self):
#         return len(self.x_files)
    
    
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.reshape(-1, 1536)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
# model = Net().to('cuda')
# summary(model, (1, 26, 34))



# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torchvision.transforms import transforms
# from torch.utils.data import DataLoader
# import torch.optim as optim

# # from data_loader import eyes_dataset
# # from model import Net

# x_train = np.load('/home/user/test_kay/test_sleepy_eyesdetection/dataset/x_train.npy').astype(np.float32)
# y_train = np.load('/home/user/test_kay/test_sleepy_eyesdetection/dataset/y_train.npy').astype(np.float32)

# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomRotation(10),
#     transforms.RandomHorizontalFlip(),
# ])

# train_dataset = eyes_dataset(x_train, y_train, transform=train_transform)

PATH = '/home/user/test_kay/test_sleepy_eyesdetection/weights/train.pth'

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# modelNet = Net()
# modelNet.to('cuda')

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(modelNet.parameters(), lr=0.0001)

# epochs = 50

# def accuracy(y_pred, y_test):
#     y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0]
#     acc = torch.round(acc * 100)
    
#     return acc

# for epoch in range(epochs):
#     running_loss = 0.0
#     running_acc = 0.0
    
#     modelNet.train()
    
#     for i, data in enumerate(train_dataloader, 0):
#         input_1, labels = data[0].to('cuda'), data[1].to('cuda')
        
#         input = input_1.transpose(1, 3).transpose(2, 3)
        
#         optimizer.zero_grad()
        
#         outputs = modelNet(input)
        
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         running_acc += accuracy(outputs, labels)
        
#         if i % 80 == 79:
#             print('epoch: [%d/%d] trian loss: %.5f train acc: %.5f' % (
#                 epoch + 1, epochs, running_loss / 80, running_acc / 80))
#             running_loss = 0.0

# print("learning finish")
# torch.save(modelNet.state_dict(), PATH)

# x_test = np.load('/home/user/test_kay/test_sleepy_eyesdetection/dataset/x_val.npy').astype(np.float32)
# y_test = np.load('/home/user/test_kay/test_sleepy_eyesdetection/dataset/y_val.npy').astype(np.float32)

# test_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# test_dataset = eyes_dataset(x_test, y_test, transform=test_transform)

# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# modelTest = Net()
# modelTest.to('cuda')
# modelTest.load_state_dict(torch.load(PATH))
# modelTest.eval()

# count = 0

# with torch.no_grad():
#     total_acc = 0.0
#     acc = 0.0
#     for i, test_data in enumerate(test_dataloader, 0):
#         data, labels = test_data[0].to('cuda'), test_data[1].to('cuda')
        
#         data = data.transpose(1, 3).transpose(2, 3)
        
#         outputs = modelTest(data)
        
#         acc = accuracy(outputs, labels)
#         total_acc += acc
        
#         count = i
#     print('average acc: %.5f' % (total_acc/count), '%')
    
# print('test finish!')


'''
check with video
'''

import cv2
import dlib
import numpy as np
import torch
from imutils import face_utils
# from model import Net

IMG_SIZE = (34, 26)
# PATH = '/home/user/test_kay/test_sleepy_eyesdetection/weights/train.pth'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/user/test_kay/test_sleepy_eyesdetection/shape_predictor_68_face_landmarks.dat')

modelDetect = Net()
modelDetect.load_state_dict(torch.load(PATH))
modelDetect.eval()

n_count = 0

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]
    
    margin_x, margin_y = w / 2, h / 2
    
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)
    
    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int32)
    
    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
    
    return eye_img, eye_rect

def predict(pred):
    pred = pred.transpose(1,3).transpose(2, 3)
    
    outputs = modelDetect(pred)
    
    pred_tag = torch.round(torch.sigmoid(outputs))
    
    return pred_tag

print("Video Start!")
cap = cv2.VideoCapture('/home/user/test_kay/test_sleepy_eyesdetection/mask_video.mp4')

if not cap.isOpened():
    print("Error! Could not open video")
    exit()

while cap.isOpened():
    ret, img_ori = cap.read()
    
    if not ret:
        break
    
    img_ori = cv2.resize(img_ori, dsize=(0,0), fx=0.5, fy=0.5)
    
    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)
        
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
        
        eye_img_l = cv2.resize(eye_img_l.astype('float32'), dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r.astype('float32'), dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        
        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
        
        eye_input_l = torch.from_numpy(eye_input_l)
        eye_input_r = torch.from_numpy(eye_input_r)
        
        pred_l = predict(eye_input_l)
        pred_r = predict(eye_input_r)
        
        if pred_l.item() == 0.0 and pred_r.item() == 0.0:
            n_count += 1
        else:
            n_count = 0
            
        if n_count > 100:
            cv2.putText(img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()