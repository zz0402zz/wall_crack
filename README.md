# 大坝裂缝检测原型

项目主线：

Label Studio -> U-Net++ (EfficientNet-B2) -> mask/overlay/report

## 目录说明

```text
data/raw/images/        原图
data/raw/masks/         从标注导出的二值掩码（与原图同名，后缀 .png）
data/processed/         训练切片数据
label_studio/           标注配置与导出文件
scripts/                入口脚本
src/dam_crack_unet/     核心代码
```

## 0. 环境准备（Windows + conda）

在项目根目录执行：

```bat
conda create -n zz python=3.10 -y
conda activate zz
python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --retries 50 --timeout 120
python -m pip install -r requirements.txt
```

建议始终使用 python -m pip，避免 pip 指向错误环境。

## 1. 生成标注任务（可选）

```bat
python scripts/make_label_studio_tasks.py
```

生成后可在 label_studio/tasks.json 查看任务。

## 2. 启动 Label Studio

项目自带的是 shell 脚本（Linux/macOS）。Windows 推荐直接运行：

```bat
label_studio start
```

在网页中新建项目后：

1. 复制并粘贴 label_studio/label_config.xml 的内容作为标注配置。
2. 导入 label_studio/tasks.json 作为任务。

## 3. 导回标注并生成 masks（关键）

把导出的 JSON 放到：

```text
label_studio/exports/picture.json
名字一定要叫picture
```

执行转换：

```bat
python scripts/import_label_studio.py --tasks label_studio/exports/picture.json --data-key image --overwrite
```

成功后应在 data/raw/masks 下看到与图片同名的 .png 文件。

## 4. 生成训练数据

默认随机划分训练/验证：

```bat
python scripts/prepare_unet_dataset.py
```

若你已手工划分原图名单：

```bat
python scripts/prepare_unet_dataset.py --train-stems-file data/splits/train_stems.txt --val-stems-file data/splits/val_stems.txt
```

## 5. 训练

```bat
python scripts/train_unetpp.py --dataset-root data/processed/dam_crack_unetpp_v1 --run-dir runs/unetpp_b2_v1 --image-size 512 --batch-size 1 --epochs 30 --num-workers 0
```

## 6. 推理

```bat
python scripts/infer_unetpp.py --checkpoint runs/unetpp_b2_v2/checkpoints/best.pt --image data/test/images/1.jpg 
```

