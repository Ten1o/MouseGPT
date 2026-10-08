# AnimateDiff的微调和推理

SWIFT已经支持了AnimateDiff的微调和推理，目前支持两种方式：全参数微调和LoRA微调。

首先需要clone并安装SWIFT：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install ".[aigc]"
```

## 全参数训练

### 训练效果

全参数微调可以复现[官方提供的模型animatediff-motion-adapter-v1-5-2](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2/summary)的效果，需要的短视频数量较多，魔搭官方复现使用了官方数据集的subset版本：[WebVid 2.5M](https://maxbain.com/webvid-dataset/)。训练效果如下：

```text
Prompt:masterpiece, bestquality, highlydetailed, ultradetailed, girl, walking, on the street, flowers
```



![image.png](../../resources/1.gif)

```text
Prompt: masterpiece, bestquality, highlydetailed, ultradetailed, beautiful house, mountain, snow top
```

![image.png](../../resources/2.gif)

2.5M子数据集训练的生成效果仍存在效果不稳定的情况，开发者使用10M数据集效果会更稳定。

### 运行命令

```shell

# Experimental environment: A100 * 4
# 200GB GPU memory totally
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --sft_type full \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 4 \
  --eval_steps 100 \
  --gradient_accumulation_steps 16
```

我们使用了A100 * 4进行训练，共需要200GB显存，训练时长约40小时。数据格式如下：

```text
--csv_path 传入一个csv文件，该csv文件应包含如下格式：
name,contentUrl
Travel blogger shoot a story on top of mountains. young man holds camera in forest.,stock-footage-travel-blogger-shoot-a-story-on-top-of-mountains-young-man-holds-camera-in-forest.mp4
```

name字段代表该短视频的prompt，contentUrl代表该视频文件的名称

```text
--video_folder 传入一个视频目录，该目录中包含了csv文件中，contentUrl指代的所有视频文件
```

使用全参数进行推理方式如下：

```shell

# Experimental environment: A100
# 18GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --sft_type full \
  --ckpt_dir /output/path/like/checkpoints/iter-xxx \
  --eval_human true
```

其中的--ckpt_dir 传入训练时输出的文件夹即可。

## LoRA训练

### 运行命令

全参数训练会从0开始训练整个Motion-Adapter结构，用户可以使用现有的模型使用少量视频进行微调，只需要运行下面的命令：

```shell

# Experimental environment: A100
# 20GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --sft_type lora \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 1 \
  --eval_steps 200 \
  --dataset_sample_size 10000 \
  --gradient_accumulation_steps 16
```

视频数据参数同上。

推理命令如下：

```shell

# Experimental environment: A100
# 18GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --sft_type lora \
  --ckpt_dir /output/path/like/checkpoints/iter-xxx \
  --eval_human true
```

其中的--ckpt_dir 传入训练时输出的文件夹即可。

## 参数列表

下面给出训练和推理分别支持的参数列表及其含义：

### 训练参数

```text
motion_adapter_id_or_path: Optional[str] = None
motion_adapter_revision: Optional[str] = None

model_id_or_path: str = None
model_revision: str = None

dataset_sample_size: int = None

sft_type: str = field(
    default='lora', metadata={'choices': ['lora', 'full']})

output_dir: str = 'output'
ddp_backend: str = field(
    default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})

seed: int = 42

lora_rank: int = 8
lora_alpha: int = 32
lora_dropout: float = 0.05
lora_dtype: str = 'fp32'

gradient_checkpointing: bool = False # Gradient checkpointing defaults to false; this diffusers version does not support true.
batch_size: int = 1 # batchsize
num_train_epochs: int = 1
# if max_steps >= 0, override num_train_epochs
learning_rate: Optional[float] = None
weight_decay: float = 0.01
gradient_accumulation_steps: int = 16
max_grad_norm: float = 1.
lr_scheduler_type: str = 'cosine'
warmup_ratio: float = 0.05

eval_steps: int = 50
save_steps: Optional[int] = None
dataloader_num_workers: int = 1

push_to_hub: bool = False
# 'user_name/repo_name' or 'repo_name'
hub_model_id: Optional[str] = None # modelhub id
hub_private_repo: bool = False
push_hub_strategy: str = field(
    default='push_best',
    metadata={'choices': ['push_last', 'all_checkpoints']})
# None: use env var `MODELSCOPE_API_TOKEN`
hub_token: Optional[str] = field(
    default=None,
    metadata={
        'help':
        'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
    })

ignore_args_error: bool = False  # True: notebook compatibility

text_dropout_rate: float = 0.1

validation_prompts_path: str = field(
    default=None,
    metadata={
        'help':
        'The validation prompts file path, use aigc/configs/validation.txt is None'
    })

trainable_modules: str = field(
    default='.*motion_modules.*',
    metadata={
        'help':
        'The trainable modules, by default, the .*motion_modules.* will be trained'
    })

mixed_precision: bool = True

enable_xformers_memory_efficient_attention: bool = True

num_inference_steps: int = 25 #
guidance_scale: float = 8.
sample_size: int = 256
sample_stride: int = 4
sample_n_frames: int = 16

csv_path: str = None
video_folder: str = None

motion_num_attention_heads: int = 8
motion_max_seq_length: int = 32
num_train_timesteps: int = 1000
beta_start: int = 0.00085
beta_end: int = 0.012
beta_schedule: str = 'linear'
steps_offset: int = 1
clip_sample: bool = False

use_wandb: bool = False
```

### 推理参数

```text
motion_adapter_id_or_path: Optional[str] = None
motion_adapter_revision: Optional[str] = None

model_id_or_path: str = None
model_revision: str = None

sft_type: str = field(
    default='lora', metadata={'choices': ['lora', 'full']})

ckpt_dir: Optional[str] = field(
    default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
eval_human: bool = False

seed: int = 42

merge_lora: bool = False # Merge lora into the MotionAdapter and save the model.
replace_if_exists: bool = False # Replace the files if the output merged dir exists when `merge_lora` is True.

# other
ignore_args_error: bool = False  # True: notebook compatibility

validation_prompts_path: str = None

output_path: str = './generated'

enable_xformers_memory_efficient_attention: bool = True

num_inference_steps: int = 25 #
guidance_scale: float = 8.
sample_size: int = 256
sample_stride: int = 4
sample_n_frames: int = 16

motion_num_attention_heads: int = 8
motion_max_seq_length: int = 32
num_train_timesteps: int = 1000
beta_start: int = 0.00085
beta_end: int = 0.012
beta_schedule: str = 'linear'
steps_offset: int = 1
clip_sample: bool = False

```
