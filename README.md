# MiniMind-V 第20轮：整图与局部切图对照（仅代码）

比较整图64视觉token与整图加四个局部裁剪、共320视觉token。顺序为整图、左上、右上、左下、右下，从原图裁剪，各视图独立缩放256；提示包含原图范围，输出原图归一化坐标。

仅包含Python、启动脚本、依赖说明及许可证，不含数据、图片、模型权重、tokenizer、日志或评测输出。原模型代码来自Apache-2.0许可的MiniMind-V，基线提交fa36e4861d08088992db3eda2636cf991ab323da，保留LICENSE。

## 环境与外部资源

Linux/Python3.10，安装适配硬件的torch2.6.0，再安装requirements.txt。GPU队列需要nvidia-smi和tmux。

运行前自行提供：

- model/siglip2-base-p32-256-ve：视觉编码器和processor。
- out/course_vqa_ctrl_r3_20261022_768.pth：原实验R3初始化，6400词表。
- dataset/course/expanded_r18_20261107/tokenizer：7400词表，两组共享；本轮答案仍为浮点坐标。
- dataset/course/expanded_r18_20261107/float_train.parquet：6844行，分类400、VQA444、检测6000依次排列，字段conversations和image_bytes。
- dataset/course/classification/val.jsonl（50条）、revision2/vqa/val.jsonl（57条）、revision2/detection/val.jsonl（40条）及各目录图片。
- dataset/course/revision2/detection/train.jsonl：原始320图训练诊断及图片。

JSONL含image_id、sample_id、image_path、conversations、question_type。分类另有label，VQA有reference_answers，检测有objects（label、bbox_normalized）。

路径默认相对仓库推导，可设MINIMIND_ROOT、R20_BASE_WEIGHT、R20_TRAIN_PARQUET、R20_TOKENIZER、R20_DETECTION_ROOT。

## 运行

在仓库根目录执行：

    tmux new -s minimind_r20
    bash scripts/course/launch_r20.sh

准备程序检查外部资源并冻结哈希，随后队列自动等待空闲GPU。每20秒检查显存≤100MiB、利用率≤5%且无计算进程，连续4次满足后启动，不停止其他任务。有两卡可并行，否则顺序运行。

两组seed42、batch4、10000步，分类25%/VQA25%/检测50%，学习率5e-6余弦降至5e-7。冻结视觉编码器，训练语言模型及投影，max_seq_len1024。每500步保存续训状态，0/2000/5000/8000/10000步自动评测，结果与权重只在运行时生成。

Ctrl+B后按D离开tmux，tmux attach -t minimind_r20恢复。再次启动队列可从latest_resume.pth续训；不要修改被冻结输入。

## 实现

model/、dataset/、trainer/保留基础模型和必要数据/训练函数。scripts/course内包含切图、训练、队列、评测及报告生成；r20_runtime、r20_model、r20_metrics_summary、r20_gpu从旧公共模块提取必要函数，去掉前19轮依赖。

这是原第20轮代码的可移植整理版，不声称整理后已重新完整训练。两组token、提示和计算量均不同，实验不是等算力比较。后续750张CPU评测属于另一个协议，本仓库不混入其代码。
