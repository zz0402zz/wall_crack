# 大坝裂缝检测原型

当前项目只保留一条主线：

`Label Studio -> U-Net++(EfficientNet-B2) -> mask/overlay/report`

## 目录

```text
data/raw/images/        原图
data/raw/masks/         标注导回后的 mask
data/processed/         训练用切片数据
label_studio/           标注配置和任务文件
scripts/                入口脚本
src/dam_crack_unet/     核心代码
```

## 核心脚本

- [label_config.xml](/Users/zz/Applications/detecion_wall/label_studio/label_config.xml)
- [make_label_studio_tasks.py](/Users/zz/Applications/detecion_wall/scripts/make_label_studio_tasks.py)
- [start_label_studio.sh](/Users/zz/Applications/detecion_wall/scripts/start_label_studio.sh)
- [import_label_studio.py](/Users/zz/Applications/detecion_wall/scripts/import_label_studio.py)
- [prepare_unet_dataset.py](/Users/zz/Applications/detecion_wall/scripts/prepare_unet_dataset.py)
- [train_unetpp.py](/Users/zz/Applications/detecion_wall/scripts/train_unetpp.py)
- [infer_unetpp.py](/Users/zz/Applications/detecion_wall/scripts/infer_unetpp.py)

## 使用流程

### 1. 生成标注任务

```bash
.venv/bin/python scripts/make_label_studio_tasks.py
```

### 2. 启动 Label Studio

```bash
./scripts/start_label_studio.sh
```

登录：

```text
email: admin@example.com
password: admin123456
```

在网页里新建项目，粘贴 [label_config.xml](/Users/zz/Applications/detecion_wall/label_studio/label_config.xml)，再导入 [tasks.json](/Users/zz/Applications/detecion_wall/label_studio/tasks.json)。

### 3. 导回标注

如果是别的电脑标好后发回来的结果，统一放到：

```text
label_studio/exports/picture.json
```

导入命令：

```bash
.venv/bin/python scripts/import_label_studio.py \
  --tasks label_studio/exports/picture.json \
  --data-key image \
  --overwrite
```

也可以继续用任意别的 JSON 文件名：

```bash
.venv/bin/python scripts/import_label_studio.py \
  --tasks /path/to/export.json \
  --data-key image \
  --overwrite
```

### 4. 生成训练数据

默认随机按原图分训练/验证：

```bash
.venv/bin/python scripts/prepare_unet_dataset.py
```

如果你已经手工分好了原图名单：

```bash
.venv/bin/python scripts/prepare_unet_dataset.py \
  --train-stems-file data/splits/train_stems.txt \
  --val-stems-file data/splits/val_stems.txt
```

### 5. 训练

```bash
.venv/bin/python scripts/train_unetpp.py \
  --dataset-root data/processed/dam_crack_unetpp_v1 \
  --run-dir runs/unetpp_b2_v1 \
  --image-size 512 \
  --batch-size 1 \
  --epochs 30 \
  --num-workers 0
```

### 6. 推理

```bash
.venv/bin/python scripts/infer_unetpp.py \
  --checkpoint runs/unetpp_b2_v1/checkpoints/best.pt \
  --image data/raw/images/example.jpg \
  --output-dir outputs/unetpp_demo
```
