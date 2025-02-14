import numpy as np
'''
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def normalize(self):
        norm = np.linalg.norm(self.to_array())
        if norm == 0:
            return self
        return Vector(self.x / norm, self.y / norm, self.z / norm)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        return t1, t2

class NormalMap:
    def __init__(self, image_path):
        self.image = self.load_image(image_path)

    def load_image(self, image_path):
        from PIL import Image
        img = Image.open(image_path)
        return np.array(img) / 255.0  # Normalize to [0, 1]

    def get_normal(self, u, v):
        # u, v are normalized coordinates in [0, 1]
        height, width, _ = self.image.shape
        x = int(u * (width - 1))
        y = int(v * (height - 1))
        normal = self.image[y, x]
        return Vector(normal[0] * 2 - 1, normal[1] * 2 - 1, normal[2])  # Convert to [-1, 1]

class Renderer:
    def __init__(self, width, height, light, objects):
        self.width = width
        self.height = height
        self.light = light
        self.objects = objects

    def render(self):
        image = np.zeros((self.height, self.width, 3))
        for y in range(self.height):
            for x in range(self.width):
                ray_direction = Vector((x / self.width) * 2 - 1, (y / self.height) * 2 - 1, 1).normalize()
                color = self.trace_ray(Vector(0, 0, 0), ray_direction)
                image[y, x] = color.to_array()
        return image

    def trace_ray(self, ray_origin, ray_direction):
        closest_t = float('inf')
        hit_object = None
        for obj in self.objects:
            t_values = obj.intersect(ray_origin, ray_direction)
            if t_values:
                for t in t_values:
                    if t and t < closest_t:
                        closest_t = t
                        hit_object = obj

        if hit_object:
            return self.calculate_color(hit_object, ray_origin, ray_direction, closest_t)
        return Vector(0, 0, 0)  # 背景颜色

    def calculate_color(self, hit_object, ray_origin, ray_direction, t):
        hit_point = ray_origin + ray_direction * t
        normal_map = hit_object.normal_map
        u, v = self.calculate_uv(hit_point)  # 计算纹理坐标
        normal = normal_map.get_normal(u, v).normalize()
        light_dir = (self.light.position - hit_point).normalize()

        # 使用法线进行光照计算
        diffuse_intensity = max(0, normal.dot(light_dir)) * self.light.intensity
        return Vector(diffuse_intensity, diffuse_intensity, diffuse_intensity)
    
    def calculate_uv(self, hit_point):
        # 这里假设简单的平面映射
        u = (hit_point.x + 1) / 2
        v = (hit_point.y + 1) / 2
        return u, v

if __name__ == "__main__":
    # 定义光源
    light_position = Vector(5, 5, 5)
    light_intensity = 1.0
    light = Light(position=light_position, intensity=light_intensity)

    # 创建球体和法线贴图
    sphere = Sphere(center=Vector(0, 0, 0), radius=1)
    sphere.normal_map = NormalMap("results/yellow_cup_0/rgb.png")  # 需要提供法线贴图路径

    # 创建渲染器
    width, height = 480, 480
    renderer = Renderer(width, height, light, [sphere])

    # 渲染图像
    image = renderer.render()

    # 保存图像
    from PIL import Image
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save('normal_mapping_image.png')

'''

import argparse
import math
import numpy as np
from scipy import ndimage
from matplotlib import pyplot
from PIL import Image, ImageOps
import os
import multiprocessing as mp
import cv2


def smooth_gaussian(im:np.ndarray, sigma) -> np.ndarray:

    if sigma == 0:
        return im

    im_smooth = im.astype(float)
    kernel_x = np.arange(-3*sigma,3*sigma+1).astype(float)
    kernel_x = np.exp((-(kernel_x**2))/(2*(sigma**2)))

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis])

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)

    return im_smooth


def gradient(im_smooth:np.ndarray):

    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.arange(-1,2).astype(float)
    kernel = - kernel / 2

    gradient_x = ndimage.convolve(gradient_x, kernel[np.newaxis])
    gradient_y = ndimage.convolve(gradient_y, kernel[np.newaxis].T)

    return gradient_x,gradient_y


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    gradient_x = ndimage.convolve(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)

    return gradient_x,gradient_y


def compute_normal_map(gradient_x:np.ndarray, gradient_y:np.ndarray, intensity=1):

    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    strength = max_value / (max_value * intensity)

    normal_map[..., 0] = gradient_x / max_value
    normal_map[..., 1] = gradient_y / max_value
    normal_map[..., 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[..., 0], 2) + np.power(normal_map[..., 1], 2) + np.power(normal_map[..., 2], 2))

    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm

    normal_map *= 0.5
    normal_map += 0.5

    return normal_map

def normalized(a) -> float: 
    factor = 1.0/math.sqrt(np.sum(a*a)) # normalize
    return a*factor

def my_gauss(im:np.ndarray):
    return ndimage.uniform_filter(im.astype(float),size=20)

def shadow(im:np.ndarray):
    
    shadowStrength = .5
    
    im1 = im.astype(float)
    im0 = im1.copy()
    im00 = im1.copy()
    im000 = im1.copy()

    for _ in range(0,2):
        im00 = my_gauss(im00)

    for _ in range(0,16):
        im0 = my_gauss(im0)

    for _ in range(0,32):
        im1 = my_gauss(im1)

    im000=normalized(im000)
    im00=normalized(im00)
    im0=normalized(im0)
    im1=normalized(im1)
    im00=normalized(im00)

    shadow=im00*2.0+im000-im1*2.0-im0 
    shadow=normalized(shadow)
    mean = np.mean(shadow)
    rmse = np.sqrt(np.mean((shadow-mean)**2))*(1/shadowStrength)
    shadow = np.clip(shadow, mean-rmse*2.0,mean+rmse*0.5)

    return shadow

def flipgreen(path:str):
    try:
        with Image.open(path) as img:
            red, green, blue, alpha= img.split()
            image = Image.merge("RGB",(red,ImageOps.invert(green),blue))
            image.save(path)
    except ValueError:
        with Image.open(path) as img:
            red, green, blue = img.split()
            image = Image.merge("RGB",(red,ImageOps.invert(green),blue))
            image.save(path)

def CleanupAO(path:str):
    '''
    Remove unnsesary channels.
    '''
    try:
        with Image.open(path) as img:
            red, green, blue, alpha= img.split()
            NewG = ImageOps.colorize(green,black=(100, 100, 100),white=(255,255,255),blackpoint=0,whitepoint=180)
            NewG.save(path)
    except ValueError:
        with Image.open(path) as img:
            red, green, blue = img.split()
            NewG = ImageOps.colorize(green,black=(100, 100, 100),white=(255,255,255),blackpoint=0,whitepoint=180)
            NewG.save(path)

def adjustPath(Org_Path:str,addto:str):
    '''
    Adjust the given path to correctly save the new file.
    '''

    path = Org_Path.split("\\")
    file = path[-1]
    filename = file.split(".")[0]
    fileext = file.split(".")[-1]

    newfilename = addto+"\\"+filename + "_" + addto + "." + fileext
    path.pop(-1)
    path.append(newfilename)

    newpath = '\\'.join(path)

    return newpath

def Convert(im, smoothness, intensity):

    # im = pyplot.imread(input_file)

    if im.ndim == 3:
        im_grey = np.zeros((im.shape[0],im.shape[1])).astype(float)
        im_grey = (im[...,0] * 0.3 + im[...,1] * 0.6 + im[...,2] * 0.1)
        im = im_grey

    im_smooth = smooth_gaussian(im, smoothness)

    sobel_x, sobel_y = sobel(im_smooth)

    normal_map = compute_normal_map(sobel_x, sobel_y, intensity)
    pyplot.imsave("Normal.png", normal_map)

    im_shadow = shadow(im)
    pyplot.imsave("AO.png",im_shadow)

    return normal_map, im_shadow

def startConvert(im):
    
    # parser = argparse.ArgumentParser(description='Compute normal map of an image')

    # parser.add_argument('input_file', type=str, help='input folder path')
    # parser.add_argument('-s', '--smooth', default=0., type=float, help='smooth gaussian blur applied on the image')
    # parser.add_argument('-it', '--intensity', default=1., type=float, help='intensity of the normal map')

    # args = parser.parse_args()

    sigma = 0
    intensity = 1
    # input_file = args.input_file
    
    # for root, _, files in os.walk(input_file, topdown=False):
    #    for name in files:
    #       input_file.append(str(os.path.join(root, name).replace("/","\\")))
    
    # if type(input_file) == str:
    normal_map, im_shadow = Convert(im, sigma, intensity)
    return normal_map, im_shadow
    # elif type(input_file) == list:
    #     for i in input_file:
    #         ctx = mp.get_context('spawn')
    #         q = ctx.Queue()
    #         p = ctx.Process(target=Convert,args=(input_file,sigma,intensity))
    #         p.start()
    #     p.join()
    
if __name__ == "__main__":
    startConvert()