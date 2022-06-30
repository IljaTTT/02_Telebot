import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

'''Процедура повышения разрешения файла lowrez.jpg в 4раза'''

'''Функция создания модели для повышения разрешения, пробовал учить сам, 
однако, по моим подсчетам, приемлего качества, удалось бы
добиться не ранее чем через неделю обучения, поэтому 
решил взять готовую модель...'''

def make_srmodel():

    model_path = 'models/RRDB_ESRGAN_x4.pth'     
    device = torch.device('cuda') 
    # device = torch.device('cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32) #Создаем модель
    model.load_state_dict(torch.load(model_path), strict=True) #Загружаем веса
    model.eval() #Выбираем режим инференса
    model = model.to(device) #Отправляем на ГПУ
    print('Модель esrgan создана')
    return model, device

'''Процедура повышения разрешения, берет lowrez.jpg, готовит тензор, считает и 
сохраняет hirez.png'''
def make_sr(model, device):

    img_path = './images/lowrez.jpg' #Входящий файл
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) #Читаем картинку
    if max(img.shape) >512:
        raise Exception('Слишком большая картинка')
    img = img * 1.0 / 255 # Готовим из нее тензор 
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()    
    img_LR = img.unsqueeze(0)
    print(f'Повышаем разрешение картинки {img_LR.shape} в 4 раза')
    img_LR = img_LR.to(device)
    

    with torch.no_grad(): #Отправляем тензор в сеть
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) #Обрабатываем выход
    output = (output * 255.0).round() 
    cv2.imwrite('./images/hirez.png', output) #Cохранем в  hirez.png'''
    return
