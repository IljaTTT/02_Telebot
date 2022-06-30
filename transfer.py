#!/usr/bin/env a4001
from __future__ import print_function
'''Смысл переноса стиля с использованием матриц Грэма в том, что мы не путаемся каким то образом 
обучить сеть для генерации изображения, вместо этого мы используем статичную, предобученную,  на большом
массиве данных сеть, пропускаем через нее изображение стиля и вычисляем некоторую его признаковую  характеристику, 
затем мы пропускаем через эту же сеть контент изображение  и вычисляем ту же характеристику, смотрим разницу, 
и пытаемся, изменяя понемногу контент изображение, добиться минимальной разницы между их признаковыми 
характеристиками. Тоесть мы как бы учим не сеть, а само контент изображение
'''
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Если есть GPU то будем работать с более детализированным изображением'''
imsize = 512 if torch.cuda.is_available() else 128  

'''Изменяем размер и преобразуем в тензор'''
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)), 
    transforms.ToTensor()])  

'''Определяем загрузчик картинки'''
def image_loader(image_name):
    image = Image.open(image_name)
    
    
    '''Убираем батч измерение, перемещаем на устройство'''
    image = loader(image).unsqueeze(0)
    
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    '''Ошибка контента. Мы отсоединяем целевой контент, для динамического 
    вычисления градиента. Он будет статичным значением а не переменной, 
    иначе метод прямого прохода ошибки бросит ошибку '''
    def __init__(self, target,):   
        super(ContentLoss, self).__init__()   
        self.target = target.detach()

    def forward(self, inp):
        self.loss = F.mse_loss(inp, self.target)
        return inp
    
def gram_matrix(input):
    '''Вычисления матриц Грэма для входа, тут a - размер батча, b - количество карт признаков
    c, d - пространственные  характеристики карт признаков'''
    a, b, c, d = input.size()     
    
    '''Изменяем размер '''
    features = input.view(a * b, c * d)
    
    '''Считаем произведение Грэма'''
    G = torch.mm(features, features.t())
    
    '''Нормализуем значения Грэм матрицы делением на количество элементов в каждой карте признаков'''
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    '''Ошибка стиля. Вычисляем ее как среднеквадратичную ошибку между произведением Грэма
    для входящего изображения и произведения Грэма для целевых признаков. Так же отсоединяем '''

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

'''Инициализируем модель как предобученную VGG19, отправляем на устройство, устанавливаем
режим инференса '''
cnn = models.vgg19(pretrained=True).features.to(device).eval()

'''Задаем статистики нормализации, характерные для картинок на которых предобучалась наша сеть, 
Обьявляем класс для нермализации'''
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()        
        '''Изменение размера тензора необходимо, чтобы мы могли напрямую выполнить
        операцию нормализации в методе forward в векторной форме'''
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


'''Функция сборки модели и функций потерь стиля и контента'''
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers, style_layers):
    '''Модуль нормализации'''
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    '''Списки лоссов'''
    content_losses = []
    style_losses = []

    '''Первый модуль нашей последовательной модели'''
    model = nn.Sequential(normalization)
    '''Тут мы собираем модель из входящей,  именуем слои '''
    i = 0  # Увеличиваем, когда видим сверточный слой
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i) 
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            '''inplace=True не очень хорошо сочетается с ContentLoss и StyleLoss'''            
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):            
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        '''Добавляем текущий модуль'''
        model.add_module(name, layer)
        '''Если текущий слой присутствует в списке content_layers то вычисляем таргет и
        контент лосс на этом слое, и добавляем в модель модуль типа ContentLoss '''
        if name in content_layers:            
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        '''Если текущий слой присутствует в списке style_layers то вычисляем таргет и
        стиль лосс на этом слое, и добавляем в модель модуль типа StyleLoss '''
        if name in style_layers:            
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    '''Теперь мы убираем все слои после последних ContentLoss и StyleLoss '''
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

'''Оптимайзер'''
def get_input_optimizer(input_img):    
    optimizer = optim.LBFGS([input_img])
    return optimizer


"""Функция переноса стиля, или генерации текстуры если texture_gen = True, 
тогда мы используем другой набор контент и стайл слоев и ждем шума в качестве контента  """
def run_style_transfer(content_img, style_img, input_img, num_steps=250,
                       style_weight=1000000, content_weight=1, texture_gen = False):
    if texture_gen:
        num_steps*=2
        content_layers_default = ['conv_1']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3 ', 'conv_4', 'conv_5',
                                'conv_6', 'conv_7','conv_8', 'conv_9', 'conv_10',
                                'conv_11', 'conv_12', 'conv_13', 
                                ]
        style_weight = 10000
        content_weight=0
        print('Создаем модель генерации текстуры')
	
    else:
        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3 ', 'conv_5']
        print('Создаем модель переноса стиля')
	
    
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        cnn_normalization_mean, cnn_normalization_std, style_img, content_img, 
        style_layers = style_layers_default, content_layers=content_layers_default)

    '''Мы оптимизируем не модель, а входное изображение'''
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Оптимизируем..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            '''Меняем изображенние'''
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    '''Ограничиваем минимальное и максимальное значение'''
    with torch.no_grad():
        input_img.clamp_(0, 1)
        
    result = input_img[0]
    

    return result

'''Функция генерации шума для генерации текстуры'''
def generate_noise(imsize=imsize):
    noise = np.random.randint(0, 255, size= (imsize//1, imsize//1, 3))
    noise = np.array(noise, dtype = 'uint8')    
    noise = Image.fromarray(noise, 'RGB')  
    noise = noise.resize((imsize, imsize))
    return noise
