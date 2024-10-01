import cv2
import numpy as np
import gradio as gr
from scipy.ndimage import map_coordinates

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

def forward_mapping_with_interpolation(image, flow):
    """
    前向映射的图像变换，结合 map_coordinates 实现插值
    :param image: 输入的彩色图像 (H, W, 3)
    :param flow: 前向映射的位移矩阵 (H, W, 2)
    :return: 插值后的变换图像
    """
    h, w, channels = image.shape
    transformed_image = np.zeros_like(image)  # 初始化目标图像 (H, W, 3)

    # 为每个通道应用相同的变换
    for c in range(channels):
        coords_y, coords_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # 计算前向映射后每个像素的目标位置
        new_x = coords_x + flow[:, :, 1]
        new_y = coords_y + flow[:, :, 0]

        # 使用 map_coordinates 进行前向映射后的插值
        transformed_channel = map_coordinates(image[:, :, c], [new_y.ravel(), new_x.ravel()], order=1, mode='nearest')
        transformed_image[:, :, c] = transformed_channel.reshape(h, w)

    return transformed_image


# 执行仿射变换
def dis(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def point_guided_deformation(image, p, q, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    ### FILL: 基于MLS or RBF 实现 image warping
    M, N, C = image.shape


    # 定义图像四个角点的坐标
    corners = np.array([[0, 0], [N-1, 0], [0, M-1], [N-1, M-1]])
    p = np.vstack([p, corners])  # 加入四个角点到 p，四角不动
    q = np.vstack([q, corners])  # 加入四个角点到 q，四角不动

    print(M, N, C)
    
    # 创建变换矩阵
    transform = np.zeros((M, N, 2), dtype=np.float32)
    alpha = 1
    p[:,[0,1]]=p[:,[1,0]]
    q[:,[0,1]]=q[:,[1,0]]
    for i in range(M):
        for j in range(N):
            # 计算权重
            w = np.array([1 / (dis(p[t], [i, j]) ** (2 * alpha)) for t in range(len(p))])
            W = np.diag(w)
            
            # 计算质心
            P = np.sum(W @ p, axis=0)
            Q = np.sum(W @ q, axis=0)
            p_star = P / np.sum(w)
            q_star = Q / np.sum(w)
            
            p_hat = p - p_star
            q_hat = q - q_star
            
            p_hat_T = np.column_stack((p_hat[:, 1], -p_hat[:, 0]))
            
            u = 0
            for k in range(len(p)):
                u += (w[k] * (q_hat[k] @ p_hat[k].T)) ** 2 + (w[k] * (q_hat[k] @ p_hat_T[k].T)) ** 2
            u = np.sqrt(u)
            
            v_star = np.array([i, j]) - p_star
            v_star_T = np.array([v_star[1], -v_star[0]])
            
            fr_v = np.zeros(2)
            for t in range(len(p)):
                A = np.dot(np.array([p_hat[t], p_hat_T[t]]), np.dot(np.array([v_star, v_star_T]).T, w[t]))
                fr_v += (1 / u) * q_hat[t] @ A
            
            L = np.sqrt(fr_v @ fr_v.T)
            l = dis([0, 0], np.array([i, j]) - p_star)
            fr_v = (l / L) * fr_v + q_star

            # 检查 NaN 值，并设置默认值
            if np.isnan(fr_v).any():
                transform[i, j, 0] = 0 - i  # 默认值
                transform[i, j, 1] = 0 - j  # 默认值
            else:
                transform[i, j, 0] = i - fr_v[0] 
                transform[i, j, 1] = j - fr_v[1]
    
    # 应用变换并显示输出图像
    #warped_image = cv2.remap(image, transform[..., 0], transform[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    warped_image = forward_mapping_with_interpolation(image,transform)

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch(share=True)
