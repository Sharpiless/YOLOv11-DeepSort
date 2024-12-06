
# **YOLOv11/8 + DeepSort 用于目标跟踪与计数**  
🚗🚶‍♂️ **使用 YOLOv11/8 和 DeepSort 实现车辆与行人实时跟踪与计数**

[![GitHub stars](https://img.shields.io/github/stars/Sharpiless/YOLOv11-DeepSort?style=social)](https://github.com/Sharpiless/YOLOv11-deepsort)  [![GitHub forks](https://img.shields.io/github/forks/Sharpiless/YOLOv11-deepsort?style=social)](https://github.com/Sharpiless/YOLOv11-DeepSort)  [![License](https://img.shields.io/github/license/Sharpiless/YOLOv11-deepsort)](https://github.com/Sharpiless/YOLOv11-DeepSort/blob/main/LICENSE)

---

## **📌 项目简介**

本项目将 **YOLOv11/8** 与 **DeepSort** 相结合，实现了对目标的实时跟踪与计数。提供了一个封装的 `Detector` 类，方便将此功能嵌入到自定义项目中。  

---

## **🚀 核心功能**

- **目标跟踪**：实时跟踪车辆与行人。
- **计数功能**：轻松统计视频流中的车辆或行人数。
- **封装式接口**：`Detector` 类封装了检测与跟踪逻辑，便于集成。
- **高度自定义**：支持训练自己的 YOLOv11/8 模型并无缝接入框架。

---

## **🔧 使用说明**

### **安装依赖**
```bash
pip install -r requirements.txt
```

确保安装了 `requirements.txt` 文件中列出的所有依赖。
### **运行 Demo**
```bash
python main_yolov11.py
python main_yolov8.py
```
---

## **🛠️ 开发说明**

### **YOLOv 检测器**

```python
class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.weights = 'weights/YOLOv11m.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes
```
- 调用 `self.detect()` 方法返回图像和预测结果
### **DeepSort 追踪器**

```python
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
```
- 调用 `self.update()` 方法更新追踪结果
---

## **📊 训练自己的模型**

训练完成后，将模型权重文件放置于 `weights` 文件夹中。

---

## **📦 API 调用**

### **初始化检测器**
```python
from AIDetector_pytorch import Detector

det = Detector()
```

### **调用检测接口**
```python
func_status = {}
func_status['headpose'] = None

result = det.feedCap(im, func_status)
```

- `im`: 输入的 BGR 图像。
- `result['frame']`: 检测结果的可视化图像。

---

## **✨ 可视化效果**

![效果图](https://img-blog.csdnimg.cn/20201231090541223.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

---

## **📚 联系作者** 
  - Bilibili: [https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)  
  - CSDN: [https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)  
  - AI Studio: [https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)  
  - GitHub: [https://github.com/Sharpiless](https://github.com/Sharpiless)  

---

## **🎉 关注我**

关注我的微信公众号，获取更多深度学习教程：  
**公众号：可达鸭的深度学习教程**  
![微信公众号二维码](https://img-blog.csdnimg.cn/20210127153004430.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

---

## **💡 许可证**

本项目遵循 **MIT License** 协议。  
**标明目标检测部分来源**：[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)