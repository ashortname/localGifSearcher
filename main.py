import gc
import time

import torch
from resnet import resnet18
import cv2
import os
from torchvision import transforms
import math
import numpy as np

from tools import do_encrypt, do_decrypt


def getGifList(root_path: str) -> list:
    """
    获取所有gif路径，需要按照实际修改
    :param root_path: Gif存储根目录
    :return:一个列表，包含所有gif完整路径
    """
    gif_list = list()
    for count in range(13, 143):
        full_path = rf"{root_path}\GIF出处第{count}期"
        if os.path.exists(full_path):
            tmp = os.listdir(full_path)
            for gif in tmp:
                if os.path.isfile(gif):
                    gif_list.append(rf"{full_path}\{gif}")
    return gif_list


# 废弃
def get_min_max_fNums_height_weight(file_list, percent_to_use=1.0):
    new_gif_list = []
    cap = cv2.VideoCapture(file_list[0])
    fNums = get_frame(cap)
    # fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # min_datas = [fNums, fHeight, fWidth, '']
    # max_datas = [fNums, fHeight, fWidth, '']
    cap.release()

    new_gif_list.append([file_list[0], fNums])
    cut = int(2)
    file_list = file_list[:int(len(file_list) * percent_to_use)]
    for gif in file_list:
        if str(gif).endswith('.gif'):
            print(f"[{cut}/{len(file_list)}]Parsing: {gif}")
            cap = cv2.VideoCapture(gif)
            fNums = get_frame(cap)
            new_gif_list.append([gif, fNums])
            # fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # # min
            # if min_datas[0] > fNums:
            #     min_datas[0] = fNums
            #     min_datas[3] = gif
            # if min_datas[1] > fHeight:
            #     min_datas[1] = fHeight
            # if min_datas[2] > fWidth:
            #     min_datas[2] = fWidth
            # # max
            # if max_datas[0] < fNums:
            #     max_datas[0] = fNums
            #     max_datas[3] = gif
            # if max_datas[1] < fHeight:
            #     max_datas[1] = fHeight
            # if max_datas[2] < fWidth:
            #     max_datas[2] = fWidth
            cap.release()
            cut += 1
    print("\n\n\n")
    return new_gif_list, None, None


# 废弃
def get_frame(gif):
    count = int(0)
    ret, frame = gif.read()
    while ret:
        ret, frame = gif.read()
        count += 1
    return count


# 废弃
def get_frame_by_path(gif):
    cap = cv2.VideoCapture(gif)
    count = int(0)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        count += 1
    cap.release()
    return count


def gif_split_to(gpath: str, fstep: int) -> list:
    """
    分割gif图片并挑选特定帧数
    :param gpath: gif路径
    :param fstep: 目标帧数
    :return: 一个列表，包含fstep个帧数
    """
    # 获取所有帧数
    frames = []
    cap = cv2.VideoCapture(gpath)
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    cap.release()
    fnum = len(frames)
    step_frame = math.ceil(fnum / fstep)
    # 防止步长大于帧数总数
    if step_frame <= 0:
        step_frame = 1
    ret = list()
    for idx in range(0, len(frames)):
        if idx % step_frame == 0 and ret:
            frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            ret.append(frame)
    # 重复最后一帧补齐到目标帧数
    while len(ret) < g_gif_need:
        ret.append(frames[len(frames) - 1])
    frames = None
    return ret


# 废弃
def gif_split_by_frame(gpath, fstep, fnum=None):
    if fnum is None:
        fnum = get_frame_by_path(gpath)
    l_img_input = []
    # 打开gif图片
    cap = cv2.VideoCapture(gpath)
    step_frame = math.ceil(fnum / fstep)
    # 防止步长大于帧数总数
    if step_frame <= 0:
        step_frame = 1
    # 计数
    idx = int(0)
    while idx <= fnum:
        ret, frame = cap.read()
        if idx == fnum:
            last_frame = frame
        if idx % step_frame == 0 and ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            l_img_input.append(frame)
        idx += 1
    # 最后一帧补齐到目标帧数
    if len(l_img_input) < g_gif_need:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum - 1)
        ret, frame = cap.read()
        while len(l_img_input) < g_gif_need:
            l_img_input.append(frame)
    cap.release()
    return l_img_input


def get_Mean(img_list: list) -> list:
    """
    返回均值
    :param img_list: 图片列表
    :return: 返回各通道均值
    """
    R_sum = 0
    G_sum = 0
    B_sum = 0
    count = len(img_list)
    for img in img_list:
        R_sum += img[:, :, 0].mean()
        G_sum += img[:, :, 1].mean()
        B_sum += img[:, :, 2].mean()
    R_mean = R_sum / count
    G_mean = G_sum / count
    B_mean = B_sum / count
    RGB_mean = [R_mean, G_mean, B_mean]
    return RGB_mean


def get_Std(img_mean: list, img_list: list) -> list:
    """
    计算图片均方差
    :param img_mean: 均值
    :param img_list: 图片列表
    :return: 各通道均方差
    """
    R_squared_mean = 0
    G_squared_mean = 0
    B_squared_mean = 0
    count = len(img_list)
    image_mean = np.array(img_mean)
    for img in img_list:
        img = img - image_mean
        R_squared_mean += np.mean(np.square(img[:, :, 0]).flatten())
        G_squared_mean += np.mean(np.square(img[:, :, 1]).flatten())
        B_squared_mean += np.mean(np.square(img[:, :, 2]).flatten())
    # 求R、G、B的方差
    R_std = math.sqrt(R_squared_mean / count)
    G_std = math.sqrt(G_squared_mean / count)
    B_std = math.sqrt(B_squared_mean / count)
    RGB_std = [R_std, G_std, B_std]
    return RGB_std


def img_to_tensor(img_list: list) -> torch.Tensor:
    """
    把图片列表转换为tensor
    :param img_list: 图片列表
    :return:tensor
    """
    img_mean = get_Mean(img_list)
    img_std = get_Std(img_mean, img_list)
    # 加载网络
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    for i in range(len(img_list)):
        img_list[i] = transform(img_list[i])
    return torch.stack(img_list, 0)


def gif_to_tensor(gpath: str, fstep: int) -> torch.FloatTensor:
    """
    gif转目标维度tensor
    :param gpath:gif路径
    :param fstep:取得目标帧数
    :return:一个tensor
    """
    list_img = gif_split_to(gpath, fstep)
    gif_tensor = img_to_tensor(list_img)
    # gif_np = [np.array(x).transpose((2, 0, 1)) for x in list_img]
    # gif_tensor = torch.FloatTensor(np.array(gif_np))
    return gif_tensor.float()


def build_database(root_path: str, percent_to_use: float = 1.0) -> None:
    """
    提取特征库
    :param root_path: gif根目录
    :param percent_to_use: 使用数据的百分比
    :return: 无返回
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # net = resnet18(pretrained=False, input_channels=3).to(device)
    # net.load_state_dict(torch.load(g_pth_path))

    # 获取gif图片路径
    gif_list = getGifList(root_path)
    gif_list = gif_list[:int(len(gif_list) * percent_to_use)]
    # gif_list, _, _ = get_min_max_fNums_height_weight(gif_list)

    l_count = int(0)
    with torch.no_grad():
        for gif_path in gif_list:
            try:
                l_count += 1
                print(f"[{l_count}/{len(gif_list)}]正在提取：{gif_path}")
                inputs = gif_to_tensor(gif_path, g_gif_need).to(device)
                out = net(inputs)
                out_np = out.cpu().numpy()
                save_name = str(gif_path).split("\\")[-1].split('.')[0]
                save_name = do_encrypt(save_name)
                # np.save(f"./tmp/{save_name}", out_np)
                np.savez_compressed(f"{g_feature_save_path}/{save_name}", a=out_np)
            except Exception as ex:
                print(f"！！！提取失败：[{gif_path}], \n {ex.__str__()}")
    gc.collect()


def load_databse() -> list:
    """
    加载本地特征库
    :return: 加载的特征库，单条数据形式为列表：[name, data]
    """
    database = list()
    database_path = g_feature_save_path
    if os.path.exists(database_path):
        ls = os.listdir(database_path)
        for tmp in ls:
            name = tmp.split('.')[0]
            name = do_decrypt(name)
            data = np.load(f"{database_path}/{tmp}")['a']
            database.append([name, data])
    return database


# 比较相似度
def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    注意有展平操作。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    """
    farr1 = arr1.ravel()
    len1 = len(farr1)
    len2 = len(arr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        arr2 = arr2[:len1]
    numer = np.sum(farr1 * arr2)
    denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(arr2 ** 2))
    similar = numer / denom
    return (similar + 1) / 2


# 比较每行相似度
def get_cosine(template, img):
    (k, n) = template.shape
    (m, n) = img.shape
    # 求模
    template_norm = np.linalg.norm(template, axis=1, keepdims=True)
    img_norm = np.linalg.norm(img, axis=1, keepdims=True)
    # 内积
    temp_img = np.dot(template, np.transpose(img))
    temp_img_norm = np.dot(np.reshape(template_norm, [k, 1]), np.reshape(img_norm, [1, m]))
    # 余弦相似度
    cos = temp_img / temp_img_norm
    return cos


def search_gif(gif_path: str, top: int = 10) -> list:
    """
    搜索 gif
    :param gif_path: gif路径
    :param top: 返回前top个数据
    :return: 包含top个信息，单条形式为：[fullpath, sim]
    """
    # 获取所有npy文件，并与目标计算相似度
    try:
        data_base = load_databse()
    except Exception as ex:
        print("Load database Error!!! \n" + ex.__str__())
    # net = resnet18(pretrained=False, input_channels=3).to(device)
    # net.load_state_dict(torch.load(g_pth_path))

    inputs = gif_to_tensor(gif_path, g_gif_need).to(device)
    with torch.no_grad():
        out = net(inputs)
        out_np = out.cpu().numpy()
    # 对比
    ret = list()
    dst_n = out_np.ravel()
    for name, img in data_base:
        try:
            sim = mtx_similar1(img, dst_n)
            # 判断是否满了
            if len(ret) == top:
                # 降序，弹出最后一个
                ret.sort(key=lambda x: x[1], reverse=True)
                ret.pop()
            ret.append([name, sim])
        except Exception as ex:
            print(ex)
    return ret


if __name__ == '__main__':
    # 基本思路是把gif的每一帧图片取出来，灰度化再拼成一个张量？
    # 把每张gif图片当成一个batch，每张图取5帧，输出五张特征图，最后进行全局平均池化得到一个张量
    # 这样的话，和分辨率就无关了？
    # 建造数据库
    # 去掉全连接层要好一些

    # 预训练模型参数保存地址
    g_pth_path = './resnet18-f37072fd.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 每张gif采集帧数，不足的按最后一帧填充
    g_gif_need = 5
    # gif本地存储路径
    g_gif_root_path = r'gif root path'
    # 特征提取保存目录
    g_feature_save_path = r'./tmp'

    # 加载网络
    net = resnet18(pretrained=False).to(device)
    net.load_state_dict(torch.load(g_pth_path))

    # """
    # # 构造本地gif特征库
    # """
    # st = time.time()
    # build_database(root_path=g_gif_root_path, percent_to_use=1.0)
    # ed = time.time()
    # print(f"提取耗时：{ed - st: .3f} s")

    """
    # 在本地库搜索gif图片
    """
    # 要搜索的目标图片地址
    dst_img = r'dst image'
    st = time.time()
    result = search_gif(dst_img, top=10)
    ed = time.time()
    print(f"搜索耗时：{ed - st: .3f} s")
    print(result)

