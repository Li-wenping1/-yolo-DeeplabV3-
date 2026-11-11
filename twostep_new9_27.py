import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
# ======================== 0. 让本地代码库优先于 pip 包 ========================
# 你的 YOLO 权重依赖 ultralytics.nn.Attmodules；该模块在你项目内的 yolov12/ultralytics 中存在，
# 但 pip 安装的 ultralytics 通常不包含它。
PROJECT_ROOT = Path(__file__).resolve().parent
YOLOV12_ROOT = PROJECT_ROOT / "yolov12"
DEEPLAB_REPO_ROOT = PROJECT_ROOT / "Deeplabv3" / "deeplabv3-plus-pytorch-main"
for _p in (YOLOV12_ROOT, DEEPLAB_REPO_ROOT):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)
# ======================== 1. 绝对路径配置（请务必修改） ========================
# YOLOv12-SLAM 模型权重文件绝对路径
YOLO_MODEL_PATH = r"D:\twostep_modle\yolov12\runs\train\exp8\weights\best.pt"
# DeepLabV3+GBC 模型权重文件绝对路径（pytorch 的 .pth 文件）
DEEPLAB_MODEL_PATH = r"D:\twostep_modle\logs\ep080-loss0.461-val_loss0.531.pth"
# 测试集原图文件夹绝对路径
TEST_IMAGES_DIR = r"D:\Data5\2"
# 测试集真实标注文件夹绝对路径（灰度图，与图像同名，后缀 .png）
TEST_LABELS_DIR = r"D:\Data5\2lab"
# 结果输出文件夹绝对路径（将自动创建）
OUTPUT_DIR = r"D:\Data6\two_222"
# ======================== 2. 实验参数 ========================
CONF_THRESH = 0.15          # 检测置信度阈值
NMS_IOU = 0.6               # NMS 的 IoU 阈值
DEEPLAB_INPUT_SIZE = (640, 640)   # 分割网络输入尺寸 (宽, 高)
NUM_CLASSES = 3             # 将在加载 Deeplab 权重后自动校准（避免与权重类别数不一致）
class_names = ['background', 'crack', 'spall']
SAVE_VISUALIZATION = True   # 是否保存可视化对比图（左：预测叠加；右：GT叠加）
SAVE_RAW_PRED_MASK = True   # 是否保存纯预测掩码（像素值为类别ID：0/1/2...）
VIS_SUBDIR = "visualizations"     # 可视化对比图输出子文件夹
MASK_SUBDIR = "pred_masks"        # 纯掩码预测图输出子文件夹
SAVE_YOLO_STAGE1 = True     # 是否保存第一阶段 YOLO 预测可视化图（带框）
YOLO_SUBDIR = "yolo_stage1"        # YOLO 可视化输出子文件夹

# ====================== 新增：YOLO前置开关 ======================
USE_YOLO_ROI =  True
# True：两阶段，YOLO检测 + ROI裁剪 + DeepLab分割
# False：单阶段，直接整张图送入DeepLab，不使用YOLO，不做ROI裁剪
# ==============================================================

# ======================== 3. 加载模型 ========================
# 假设 YOLOv12-SLAM 使用 Ultralytics YOLO 接口
try:
    # 如果之前意外导入了 pip 的 ultralytics，清掉缓存，确保走本地 yolov12/ultralytics
    if "ultralytics" in sys.modules:
        del sys.modules["ultralytics"]
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)
# 加载 DeepLabV3+GBC（PyTorch 模型）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
def load_deeplab_model(weight_path, device):
    """按本仓库 Deeplabv3-plus-pytorch-main 的方式加载（构建网络 + load_state_dict）。"""
    def _extract_state_dict(obj):
        # 兼容：直接 state_dict / checkpoint(dict) / 其它保存格式
        if isinstance(obj, dict):
            for key in ("state_dict", "model", "model_state_dict", "net", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]
            # 很多训练脚本就是直接 torch.save(model.state_dict())
            if all(isinstance(k, str) for k in obj.keys()):
                return obj
        return obj
    def _strip_module_prefix(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        if any(k.startswith("module.") for k in state_dict.keys()):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Deeplab 权重不是 state_dict 格式。请确认该 .pth 是用 torch.save(model.state_dict()) 保存的。"
        )
    # 从最后分类卷积层推断 num_classes（比手写常量更稳）
    inferred_num_classes = None
    for k, v in state_dict.items():
        if k.endswith("cls_conv.weight") and hasattr(v, "shape") and len(v.shape) >= 1:
            inferred_num_classes = int(v.shape[0])
            break
    if inferred_num_classes is None:
        inferred_num_classes = 4  # 回退到该 repo 默认值
    # 参考 Deeplabv3-plus-pytorch-main/deeplab.py
    from nets.deeplabv3_plus import DeepLab
    # 这里 backbone/downsample_factor 必须与训练时一致，否则可能 load_state_dict 失败
    backbone = "mobilenet"
    downsample_factor = 8
    model = DeepLab(
        num_classes=inferred_num_classes,
        backbone=backbone,
        downsample_factor=downsample_factor,
        pretrained=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(
        f"DeepLab model loaded. num_classes={inferred_num_classes}, "
        f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
    )
    return model, inferred_num_classes
deeplab_model, _deeplab_num_classes = load_deeplab_model(DEEPLAB_MODEL_PATH, device)
# 用 Deeplab 的类别数校准评估配置（防止 logits 通道数与 NUM_CLASSES 不一致）
NUM_CLASSES = int(_deeplab_num_classes)
if NUM_CLASSES == 4:
    class_names = ["_background_", "spall", "rebar", "crack"]
else:
    class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
# 标注编码（你的数据集定义）：0=background, 1=spall, 2=rebar, 3=crack
BACKGROUND_ID = 0
SPALL_ID = 1
REBAR_ID = 2
CRACK_ID = 3
# ======================== 4. 辅助函数 ========================
def compute_metrics(pred_mask, gt_mask, num_classes):
    """全局指标计算"""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    cm = confusion_matrix(gt_flat, pred_flat, labels=list(range(num_classes)))
    
    class_iou = []
    class_precision = []
    class_recall = []
    class_f1 = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_iou.append(iou)
        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)
    
    # 与 Deeplab 仓库的 compute_mIoU 口径一致：包含 background
    mIoU = float(np.mean(class_iou))
    mPA = float(np.mean(class_recall))
    avg_precision = float(np.mean(class_precision))
    avg_recall = float(np.mean(class_recall))
    avg_f1 = float(np.mean(class_f1))
    return mIoU, avg_precision, avg_recall, avg_f1, mPA, class_iou
def fast_hist(a, b, n):
    """与 Deeplabv3-plus-pytorch-main/utils/utils_metrics.py 对齐的 hist 统计。"""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1e-12)
def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1e-12)
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1e-12)
def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)
def deeplab_inference(crop_img, model, input_size, device):
    """对裁剪图进行分割，返回与原裁剪区域相同尺寸的掩码"""
    def _letterbox_rgb(image_rgb, size_wh, pad_value=128):
        """等比例缩放 + 灰条填充，匹配该 Deeplab 仓库的 resize_image 行为。"""
        target_w, target_h = size_wh  # (w, h)
        ih, iw = image_rgb.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, dx, dy, nw, nh
    # 训练/官方预测使用 RGB；OpenCV 读入为 BGR，这里统一转 RGB
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    input_rgb, dx, dy, nw, nh = _letterbox_rgb(crop_rgb, input_size)
    # 预处理：转为 Tensor，归一化到 [0,1]
    tensor = torch.from_numpy(input_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)   # [1, C, H, W]
        pred_full = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    # 去掉 padding，再缩放回裁剪原始尺寸
    pred_nopad = pred_full[dy:dy + nh, dx:dx + nw]
    pred_mask = cv2.resize(
        pred_nopad,
        (crop_img.shape[1], crop_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return pred_mask

# ===================== 修改函数，增加 use_yolo 开关 =====================
def two_step_inference(image_path, yolo_model, deeplab_model, conf_thresh, nms_iou, deeplab_input_size, device, use_yolo=True, yolo_vis_save_path=None):
    """单图推理
    use_yolo=True：两阶段 YOLO检测+ROI扩边裁剪+DeepLab分割
    use_yolo=False：单阶段，整张原图直接送入DeepLab，不调用YOLO
    蓝色框：原始YOLO预测框（带类别+置信度）
    红色框：向外扩张10%的ROI框（用于裁剪送入分割网络）
    """
    # 读取原图
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = img.shape[:2]
    global_mask = np.zeros((h, w), dtype=np.uint8)

    if not use_yolo:
        # ========== 单阶段分支：整张图直接DeepLab推理 ==========
        global_mask = deeplab_inference(img, deeplab_model, deeplab_input_size, device)
        return global_mask
    
    # ========== use_yolo = True，原始两阶段逻辑 ==========
    # 检测
    results = yolo_model(image_path, conf=conf_thresh, iou=nms_iou)
    boxes = results[0].boxes

    # ========== 扩边比例，消融实验直接修改这里 0.0 /0.1 /0.2 ==========
    expand_ratio =0.2
    if boxes is None or len(boxes) == 0:
        return global_mask  # 全背景
    
    # 保存 YOLO 第一阶段预测可视化图：原图上同时绘制【原始蓝框+类别概率】和【扩边红框】
    vis_img = None
    if yolo_vis_save_path is not None:
        try:
            yolo_vis_save_path = Path(yolo_vis_save_path)
            yolo_vis_save_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img = img.copy()
        except Exception as e:
            print(f"[Warn] YOLO可视化准备失败: {image_path} -> {e}")
            vis_img = None
    # 按置信度降序排列
    order = boxes.conf.argsort(descending=True).cpu().numpy()
    for idx in order:
        box = boxes.xyxy[idx].cpu().numpy()
        cls_id = int(boxes.cls[idx].cpu().numpy())
        conf = float(boxes.conf[idx].cpu().numpy())
        x1, y1, x2, y2 = box
        # ========== 计算扩边后的ROI ==========
        box_w = x2 - x1
        box_h = y2 - y1
        delta_x = box_w * expand_ratio
        delta_y = box_h * expand_ratio
        new_x1 = x1 - delta_x
        new_y1 = y1 - delta_y
        new_x2 = x2 + delta_x
        new_y2 = y2 + delta_y
        # 边界截断，不能超出图片范围
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
        x1_exp, y1_exp, x2_exp, y2_exp = map(int, [new_x1, new_y1, new_x2, new_y2])
        if x2_exp <= x1_exp or y2_exp <= y1_exp:
            continue
        crop = img[y1_exp:y2_exp, x1_exp:x2_exp]
        if crop.size == 0:
            continue
        sub_mask = deeplab_inference(crop, deeplab_model, deeplab_input_size, device)
        # 把分割结果放回全局掩码（使用扩边框坐标）
        global_mask[y1_exp:y2_exp, x1_exp:x2_exp] = sub_mask
        # ========== 可视化绘制 ==========
        if vis_img is not None:
            # 蓝色：原始YOLO框
            b_x1, b_y1, b_x2, b_y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(vis_img, (b_x1, b_y1), (b_x2, b_y2), (255, 180, 0), 2)
            # 类别+置信度文字
            text = f"{yolo_model.names[cls_id]} {conf:.2f}"
            cv2.putText(vis_img, text, (b_x1, max(b_y1-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,0), 2)
            # 红色：扩边后的ROI框
            cv2.rectangle(vis_img, (x1_exp, y1_exp), (x2_exp, y2_exp), (0,0,255), 2)
    # 保存可视化图
    if yolo_vis_save_path is not None and vis_img is not None:
        cv2.imwrite(str(yolo_vis_save_path), vis_img)
    return global_mask

# ======================== 5. 批量评估主函数 ========================
def evaluate_two_step():
    # 创建输出目录
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / VIS_SUBDIR
    mask_dir = out_dir / MASK_SUBDIR
    yolo_dir = out_dir / YOLO_SUBDIR
    if SAVE_VISUALIZATION:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_RAW_PRED_MASK:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_YOLO_STAGE1:
        yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有测试图像（支持 .jpg, .png）
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_paths.extend(Path(TEST_IMAGES_DIR).glob(ext))
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        print(f"错误：在 {TEST_IMAGES_DIR} 中未找到图片文件。")
        return
    
    results = []
    # 与 Deeplab 官方评估一致：累加整个测试集的混淆矩阵，再一次性计算指标
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img_path in tqdm(image_paths, desc="Processing"):
        # 寻找对应的标注文件（同名 .png）
        label_path = Path(TEST_LABELS_DIR) / (img_path.stem + '.png')
        if not label_path.exists():
            print(f"警告：标注 {label_path} 不存在，跳过 {img_path.name}")
            continue
        
        # 读取真实标注（灰度图，像素值 0,1,2...）
        gt_mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"警告：无法读取标注 {label_path}")
            continue
        
        # 推理调用，传入USE_YOLO_ROI开关
        yolo_vis_path = (yolo_dir / f"{img_path.stem}_yolo.jpg") if (SAVE_YOLO_STAGE1 and USE_YOLO_ROI) else None
        pred_mask = two_step_inference(
            str(img_path),
            yolo_model,
            deeplab_model,
            CONF_THRESH,
            NMS_IOU,
            DEEPLAB_INPUT_SIZE,
            device,
            use_yolo=USE_YOLO_ROI,
            yolo_vis_save_path=yolo_vis_path,
        )
        
        # 计算指标（确保尺寸一致）
        if pred_mask.shape != gt_mask.shape:
            print(f"尺寸不一致：{img_path.name}, 预测 shape {pred_mask.shape}, 标注 shape {gt_mask.shape}")
            continue
        
        # 累计全局 hist（Deeplab 口径）
        hist += fast_hist(gt_mask.flatten(), pred_mask.flatten(), NUM_CLASSES)
        # 逐图指标仍可保留到 CSV（注意：这是“逐图平均口径”，与 Deeplab 的全局 hist 口径不同）
        mIoU, prec, rec, f1, mPA, class_iou = compute_metrics(pred_mask, gt_mask, NUM_CLASSES)
        
        results.append({
            'image': img_path.name,
            'USE_YOLO_ROI': USE_YOLO_ROI,
            'mIoU': mIoU,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'mPA': mPA,
            'class_iou_background': class_iou[0],
            'class_iou_crack': class_iou[CRACK_ID] if NUM_CLASSES > CRACK_ID else 0,
            'class_iou_spall': class_iou[SPALL_ID] if NUM_CLASSES > SPALL_ID else 0,
        })
        # 保存纯预测掩码（不叠加）
        # 保存纯预测掩码（灰度ID图）
        if SAVE_RAW_PRED_MASK:
            cv2.imwrite(str(mask_dir / f"{img_path.stem}_pred_mask.png"), pred_mask)
            # ========== 新增：生成彩色掩码图 ==========
            color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            # 0 background: black
            color_mask[pred_mask == SPALL_ID] = [0, 255, 255]   # spall yellow BGR
            color_mask[pred_mask == REBAR_ID] = [0, 0, 255]     # rebar red BGR
            color_mask[pred_mask == CRACK_ID] = [255, 255, 255] # crack white BGR
            cv2.imwrite(str(mask_dir / f"{img_path.stem}_color_pred_mask.png"), color_mask)
        
        # 保存可视化对比图（左：原图+预测叠加；右：原图+GT叠加）
        if SAVE_VISUALIZATION:
            vis_img = cv2.imread(str(img_path))
            if vis_img is None:
                print(f"警告：无法读取原图用于可视化：{img_path}")
            else:
                # 颜色（OpenCV 为 BGR）：spall=黄, rebar=红, crack=白
                overlay_pred = np.zeros_like(vis_img, dtype=np.uint8)
                if NUM_CLASSES > SPALL_ID:
                    overlay_pred[pred_mask == SPALL_ID] = [0, 255, 255]
                if NUM_CLASSES > REBAR_ID:
                    overlay_pred[pred_mask == REBAR_ID] = [0, 0, 255]
                if NUM_CLASSES > CRACK_ID:
                    overlay_pred[pred_mask == CRACK_ID] = [255, 255, 255]
                vis_pred = cv2.addWeighted(vis_img, 0.6, overlay_pred, 0.4, 0)
                overlay_gt = np.zeros_like(vis_img, dtype=np.uint8)
                if NUM_CLASSES > SPALL_ID:
                    overlay_gt[gt_mask == SPALL_ID] = [0, 255, 255]
                if NUM_CLASSES > REBAR_ID:
                    overlay_gt[gt_mask == REBAR_ID] = [0, 0, 255]
                if NUM_CLASSES > CRACK_ID:
                    overlay_gt[gt_mask == CRACK_ID] = [255, 255, 255]
                vis_gt = cv2.addWeighted(vis_img, 0.6, overlay_gt, 0.4, 0)
                compare = np.concatenate([vis_pred, vis_gt], axis=1)
                cv2.imwrite(str(vis_dir / f"{img_path.stem}_compare.png"), compare)
    
    if len(results) == 0:
        print("没有成功处理任何图像。")
        return
    
    # 汇总统计
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "two_step_metrics.csv", index=False)
    
    # ======================== Deeplab 对齐口径（全局 hist） ========================
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    Accuracy = per_Accuracy(hist)
    F1s = (2 * Precision * PA_Recall) / np.maximum((Precision + PA_Recall), 1e-12)
    print("\n========= Evaluation Results (Deeplab-aligned) =========")
    print(f"USE_YOLO_ROI = {USE_YOLO_ROI}")
    print(f"Total images: {len(df)}")
    print(f"mIoU: {float(np.nanmean(IoUs)):.4f}")
    print(f"mPA(Recall): {float(np.nanmean(PA_Recall)):.4f}")
    print(f"mPrecision: {float(np.nanmean(Precision)):.4f}")
    print(f"mF1: {float(np.nanmean(F1s)):.4f}")
    print(f"Accuracy: {float(Accuracy):.4f}")
    # 逐类输出（对齐 utils_metrics.py 的风格）
    for i in range(NUM_CLASSES):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(
            f"===> {name}: IoU-{IoUs[i]:.4f}; Recall(mPA)-{PA_Recall[i]:.4f}; Precision-{Precision[i]:.4f}"
        )
    print("==================================================================================")
    # 额外保存一份 Deeplab 口径的全局混淆矩阵，方便对照
    pd.DataFrame(hist).to_csv(out_dir / "confusion_matrix_global.csv", index=False)
if __name__ == "__main__":
    evaluate_two_step()
