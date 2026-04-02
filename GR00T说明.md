# GR00T 概览(4.29最新版)
目前最新训练的效果最好的模型在H200上，H200如何连接请参考这里  
http://192.168.0.60:50080/robotbrain/proxymanual/-/blob/main/服务器信息.md  
如果有问题，可以请教张老师

~~deprecated：  
4.16采集的100个episdoe已经清洗好了，模型就是按照清洗好的数据做训练的．目前4.16有2版数据用于训练，一版是最新几次采集的300个episode，另一版是最新几次的200个episode．目前用这2版本数据分别在H200训练了2个模型，具体可以使用不同checkpoint，对比下哪个模型效果好．
数据清洗+模型训练工作目前还没做总结，之后如果4.16的模型效果还行，再考虑标准化的数据清洗链路+训练文档．之前微调过的模型都放在了路径/home/YY/tmp/下面，建议根据名称里的日期，用合适的checkpoint(可以试试最新的4.16的2个模型，4.15的目前仅能抓住水瓶，但无法抬起)~~

目前最新的任务Task：  
让FZWX机器人抓取粉色泡沫块，然后放到黑色板子上

目前最新使用的数据：  
4.23采集的200个episode数据，已经清洗好了，路径在H200服务器上的/home/YY/Datasets/GR00T/FZWX_0423_train_FOAM

用上述数据训练出的最新的2版模型：  
模型1：/home/YY/tmp/fzwx-checkpoints-0423-Foam-3K-CuoKai  
模型2：/home/YY/tmp/fzwx-checkpoints-0423-Foam-50K-CuoKai  

模型说明：
- 其中模型1共训练了3k轮，模型共训练了50k轮，目前这两版中各自最好的模型都可用较高准确率实现任务
- 3k最好的模型的抓取速度比50k最好的模型要慢
- 最终我们挑选最好的模型为模型2中的第10k轮的checkpoint模型
- 因为我们本次任务采集的都是右臂抓取任务，所以目前模型只输出右臂7+1=8自由度的action

直接按照下面Step 1和Step 2操作就可以实机运行目前微调过的GR00T

## Step 1: 在H200起微调后的模型服务
主服务的代码是用H200服务器上的脚本
/home/YY/Isaac-GR00T/scripts/inference_service.py
起来的．先在H200上面激活conda环境gr00t（官方gr00t的github环境）：
```
conda activate gr00t
```
然后进入路径
```
cd /home/YY/Isaac-GR00T
```

在这个路径下，运行下面的命令(找张显存没太满的单卡就能启动模型服务，例如在第6张卡上启动效果最好的10k的checkpoint模型)：
```
CUDA_VISIBLE_DEVICES=6 python scripts/inference_service.py --server \
    --model_path /home/YY/tmp/fzwx-checkpoints-0423-Foam-50K-CuoKai/checkpoint-10000 \
    --embodiment_tag new_embodiment \
    --port 51807 \
    --data_config fzwx \
    --denoising_steps 8
```
- --model_path参数指定微调后的checkpoint路径．例如上面是04.23训练的第10000轮时候的checkpoint
- --port指定H200上该模型服务的端口，只要这个端口51807没被占用，这里可以写死为这个．H200服务器可用端口范围参见http://192.168.0.60:50080/robotbrain/proxymanual/-/blob/main/服务器信息.md#h200
- --denoising_steps这个参数是模型推理时候，快系统diffusion模型部分推理时的denoising步数，GR00T官方给的4，这里改成了8，试过4 8 16，看起来这个参数不是特别敏感，可以尝试4~32这些范围试试看效果
- 其他参数写死的，不用动



## Step 2: 公司5层4090服务器客户端
按照上面步骤在H200启动好模型服务之后，登陆到公司192.168.0.104的服务器上(连接ZJUROBOT无线网即可ssh登陆)，建议先验证该4090服务器和H200服务器的连通性：
```
ping 192.168.88.230
```
如果连通性没问题，进入4090服务器上docker容器minicpm_yy2，该容器已经正在运行，无需启动，直接进入
```
docker exec -it minicpm_yy2 bash
```
然后进入里面的路径：
```
cd /home/Isaac-GR00T/getting_started
```
然后conda激活lerobot环境（官方lerobot的github环境）：
```
conda activate lerobot
```
然后执行命令：
```
python eval_gr00t_FZWX.py --use_policy \
    --actions_to_execute 600 \
    --host 192.168.88.230 \
    --port 51807 \
    --action_horizon 12
```
- --use_policy写死，不要动
- --actions_to_execute指的执行动作多少个time step，这里我们采集数据时的视频大概600-800帧左右，这里就写成了600．这个参数也可以写的大点，不过运行起来一般不会让都跑完，因为比较慢．除非首次抓取失败，多给step来等待，或者需要重复抓取的情况
- --host 192.168.88.230 --port 51807，这个则是刚才步骤H200服务器IP地址和模型服务端口
- --action_horizon参数控制一次推理生成接下来的16个(这里16是GR00T使用的，可以不用管)动作，只用前action_horizon个．现在12是GR00T的github默认使用的．最大16，建议不要超过12．不过小的4没试过，可以试试

**:warning: :warning: :warning: 注意事项(必读)：**
1. 这个客户端代码eval_gr00t_FZWX.py理论上可以在不同设备上使用，只要能调用到H200的模型服务就行．这里目前选择放在了公司5楼的服务器上．延迟和慢的问题暂未做处理
2. 运行代码eval_gr00t_FZWX.py需要小心，里面的set_target_state和send_action逻辑是直接下发操控FZWX机器人的．**不排除会出现极端模型预测的action情况，导致机器人出现大的动作，执行该代码时，请全程观察机器人的动作，如果遇到危险动作，请立即ctrl+c杀掉该代码**
3. eval_gr00t_FZWX.py里的move_to_initial_pose，写死了FZWX机器人的初始零位，即每次从这个零位开始实机运行抓取任务
4. 如果需要更快的频率，请减小目前代码里的time.sleep(0.01)，目前0.01就是GR00T官方例子给的时间
5. eval_gr00t_FZWX.py里的参数--lang_instruction目前写死的Pick up the pink block and place it on the left black square.因为微调时也用的这个指令
6. 在运行eval_gr00t_FZWX.py进行实机推理时
    - 在客户端路径/home/Isaac-GR00T/getting_started下面会生成remote_images和remote_data的2个路径，里面存放实时的数据和图像(图像只有抬头的摄像头)
    - 另外，在/home/Isaac-GR00T/getting_started/下面还会实时刷新2张图像output_image_right.jpg和output_image.jpg，分别是实时的摄像头和右臂爪上的摄像头．建议在跑客户端代码时同时打开这2张图片，观察是否它们和机器人当前情况一致


## 附录：原hdf5清洗出模型训练数据
具体的数据清洗脚本示例都在H200的/home/YY/Datasets/GR00T/FZWX_0423_train_FOAM下面，大致步骤：
- 用脚本hdf5_to_video.sh，hdf5抽取mpeg格式的头顶+右臂mp4视频，然后转化为h264格式（这个格式支持官方gr00t模型，其他格式未做尝试）
- 上面的.sh脚本需要用到py文件visualize_episodes_v1.py，这个是基于冯老师之前给的代码稍做了改动．代码里的一些路径参数，如果需要调整，请自行改动
- 利用GenSchema_main.ipynb，仅用该notebook里的＂制作parquet 文件＂和＂制作episodes.jsonl＂，制作gr00t需要的数据schema格式．它们分别会生成一些模型训练文件和配置文件，存储在训练数据的meta路径下面
- 利用open_analysis_par.ipynb修改action和state错开








