import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Any, Optional

from models_new_version_run.VQ_VAE import IMU_VQ_Model

try:
    import yaml
except Exception:
    yaml = None


# --task_name=alignment
# --is_training=1
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
# --root_path="D:/fuy/MyCode/SensorLLMLib/datasets/MHEALTHDATASET"
# --model_id=MHealth
# --run_id="alignment_weight"
# --datasets=MHealth
# --model="Alignment_Stage"
# --data=MHealth
# --dataset_key=mealth
# --seq_len=100
# --patch_len=50
# --stride=50
# --stage=1
# --batch_size=16
# --llama_name="D:\fuy\MyCode\Llama-3.2-1B"
# --learning_rate=0.001
# --train_epochs=10
# --num_workers=0
# --vqvae_path=qua_recon_path
# --test_subjects="subject1,subject3,subject6"

class AlignmentModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.stage = int(getattr(args, "stage", 1))
        self.device = args.device

        print(f"[Model] Init in Stage: {self.stage}")

        # 1. 解析 Stage
        self.stage = int(getattr(args, "stage", 1))
        print(f"init Model in Stage: {self.stage}")
        # -------- dataset cfg load --------
        self.dataset_key = str(getattr(args, "dataset_key", getattr(args, "data", "mhealth"))).lower()
        self.ds_cfg: Dict[str, Any] = {}
        if hasattr(args, "ds_cfg") and isinstance(args.ds_cfg, dict):
            self.ds_cfg = args.ds_cfg
        else:
            ts_yaml = getattr(args, "ts_backbone_yaml", None)
            if ts_yaml is not None:
                if yaml is None:
                    raise ImportError("pyyaml not installed but ts_backbone_yaml is set.")
                if not os.path.exists(ts_yaml):
                    raise FileNotFoundError(ts_yaml)
                with open(ts_yaml, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                if self.dataset_key not in cfg_all:
                    raise KeyError(f"{self.dataset_key} not in {ts_yaml}")
                self.ds_cfg = cfg_all[self.dataset_key]
        self.device = args.device
        self.C = int(self.ds_cfg.get("channel_num", getattr(args, "enc_in", 15)))
        self.num_class = int(getattr(args, "num_class", 12))
        self.seq_len_orig = int(getattr(args, "seq_len", 200))  # 原始数据长度

        vqvae_path = getattr(args, "vqvae_path", None)
        alignment_path = getattr(args, "alignment_path", None)
        self.qua_path = self.ds_cfg.get(vqvae_path, None)
        self.vq_net = IMU_VQ_Model(args)

        if self.qua_path is not None:
            vq_net_state_dict = torch.load(self.qua_path + os.sep + "best_wrapper.pth", map_location='cpu')
            if 'state_dict' in vq_net_state_dict:
                vq_net_state_dict = vq_net_state_dict['state_dict']
            self.vq_net.load_state_dict(vq_net_state_dict, strict=True)
            self.vq_net.eval()  # Set the model to evaluation mode

            # set vq net requires_grad to False
            for param in self.vq_net.parameters():
                param.requires_grad = False
            # --------------------------------------------
        # --- 2. Calculate Alignment (Critical) ---
        # VQ Stride = stride_t ^ down_t (e.g., 2^3 = 8)
        self.vq_stride = self.vq_net.stride_t ** self.vq_net.down_t
        self.patch_len = self.vq_stride  # Student Patch MUST match VQ Stride

        # Calculate Padded Length
        # e.g., if L=195, Stride=8 -> L_pad=200
        self.seq_len_pad = ((self.seq_len_orig + self.vq_stride - 1) // self.vq_stride) * self.vq_stride
        self.P = self.seq_len_pad // self.patch_len

        print(
            f"[Model Alignment] Orig={self.seq_len_orig}, Stride={self.vq_stride} -> Padded={self.seq_len_pad}, Patches(P)={self.P}")

        # --- 3. Initialize Student ---
        self.num_primitives = self.vq_net.code_num
        self.mask_token_id = self.num_primitives
        self.dim_student = int(getattr(args, "dim_student", 256))
        if hasattr(self, 'qua_path') and self.qua_path:
            cb_path = self.qua_path + os.sep + "best_codebook.pth"
            if os.path.exists(cb_path):
                self.codebook = torch.load(cb_path, map_location="cpu").to(self.device)
            else:
                self.codebook = self.vq_net.quantizer.codebook.data.to(self.device)
        else:
            self.codebook = self.vq_net.quantizer.codebook.data.to(self.device)



        # 获取 VQ Codebook 用于初始化
        # 假设 codebook shape: [num_codes, vq_dim]
        self.num_vq_codes = self.codebook.shape[0]
        self.vq_dim = self.codebook.shape[1]

        # 2. 拿来 Llama (冻结)
        print(f"[Align] Loading Frozen Llama from {args.llama_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            args.llama_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device).eval()
        for p in self.llm.parameters(): p.requires_grad = False
        self.llm_dim = self.llm.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(args.llama_name)

        # 3. 定义要训练的层 (Adapters)
        # Projector: VQ Space -> LLM Space
        self.projector = nn.Linear(self.vq_dim, self.llm_dim).to(self.device)

        # Head: LLM Space -> VQ Space (预测下一个 Token)
        self.output_head = nn.Linear(self.llm_dim, self.num_vq_codes).to(self.device)

        # 初始化建议：Projector 最好稍微对齐一点，而不是完全随机
        # 但如果是 Linear，Xavier 初始化即可
        nn.init.xavier_normal_(self.projector.weight)
        nn.init.xavier_normal_(self.output_head.weight)

        # --- [新增] System Prompt 初始化 (与 Teacher 保持完全一致) ---
        self.system_prompt = "Analyze sensor sequence:"
        self.prompt_input_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(self.device)

        # 预计算 Prompt Embeddings (冻结)
        with torch.no_grad():
            self.prompt_embeds = self.llm.get_input_embeddings()(self.prompt_input_ids)
            # self.prompt_embeds shape: [1, L_text, D]

    def forward(self, x_imu, padding_mask=None, mode=None, labels=None):
        """
        x_imu: [B, L, C]
        """
        B = x_imu.shape[0]

        with torch.no_grad():
            gt_ids = self.vq_net.get_token_ids(x_imu)  # [B, T]
            vq_embeds = F.embedding(gt_ids, self.codebook)  # [B, T, D_vq]

        # 1. Project VQ features
        sensor_embeds = self.projector(vq_embeds)  # [B, T, D_llm]

        # 2. [关键] 拼接 Prompt + Sensor
        # batch_prompt: [B, L_text, D_llm]
        batch_prompt = self.prompt_embeds.expand(B, -1, -1)

        # inputs_embeds: [B, L_text + T, D_llm]
        inputs_embeds = torch.cat([batch_prompt, sensor_embeds], dim=1).to(self.llm.dtype)

        # 3. Feed to LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        # 4. [关键] 切片 (Slicing) - 拿掉 Prompt 部分的输出
        # 我们不需要预测 Prompt，也不需要基于 Prompt 预测第一个 Sensor Token (通常 NTP 从第一个有效 Token 开始)
        # 或者为了简单，我们让 Output Head 只处理 Sensor 部分的 hidden states
        L_text = batch_prompt.shape[1]
        sensor_hidden = last_hidden[:, L_text:, :]  # [B, T, D_llm]

        # 5. Predict Logits
        logits = self.output_head(sensor_hidden.float())  # [B, T, num_codes]

        # 6. Calculate Loss (Next Token Prediction)
        # logits[t] 预测 gt_ids[t+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = gt_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.num_vq_codes),
            shift_labels.view(-1)
        )

        return loss,None,None


    #  另一个版本的loss
    '''
        def forward(self, x_imu, padding_mask=None, mode=None, labels=None):
        B = x_imu.shape[0]

        # 1. 获取 GT (保持不变)
        with torch.no_grad():
            gt_ids = self.vq_net.get_token_ids(x_imu)
            vq_embeds = F.embedding(gt_ids, self.codebook)

        # 2. Projector & LLM (保持不变)
        sensor_embeds = self.projector(vq_embeds)
        batch_prompt = self.prompt_embeds.expand(B, -1, -1)
        inputs_embeds = torch.cat([batch_prompt, sensor_embeds], dim=1).to(self.llm.dtype)

        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        # 3. 切片 & 预测 Logits (保持不变)
        L_text = batch_prompt.shape[1]
        sensor_hidden = last_hidden[:, L_text:, :]
        logits = self.output_head(sensor_hidden.float())  # [B, T, K]

        # --- Loss 1: Cross Entropy (你原本的) ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = gt_ids[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, self.num_vq_codes),
            shift_labels.view(-1)
        )

        # --- [新增] Loss 2: Reconstruction Loss (物理约束) ---
        # 技巧：使用 Softmax 得到概率分布，进行加权求和，使其可导
        # shift_logits 预测的是 gt_ids[..., 1:] (即 t+1 时刻的动作)

        # a. 计算软分布 (Gumbel Softmax 或 Softmax)
        probs = F.softmax(shift_logits, dim=-1)  # [B, T-1, K]

        # b. "软"查表: 用概率加权 Codebook
        # [B, T-1, K] @ [K, D] -> [B, T-1, D]
        z_q_pred = torch.matmul(probs, self.codebook)

        # c. 解码: 这一步需要你的 vq_net 暴露 decoder
        # 注意: 你的 x_imu 需要切掉第一个时间步，与 shift_labels 对齐
        # x_imu_target: [B, T-1, C]
        # 注意: 这里通常需要对齐长度，假设 x_imu 已经被 Pad 好了
        # 为了简单，我们可以只计算 Latent 层的 MSE，避免调用 Decoder (省显存)

        # 方案 A (推荐): Latent Consistency Loss (不需要 Decoder)
        # 直接比较 "预测的向量" 和 "真实的 GT 向量"
        gt_z_q = vq_embeds[:, 1:, :]  # [B, T-1, D] (GT 的 t+1 时刻向量)
        loss_mse = F.mse_loss(z_q_pred, gt_z_q)

        # 方案 B (论文做法): End-to-End Reconstruction (需要 Decoder)
        # x_recon = self.vq_net.decoder(z_q_pred)
        # loss_mse = F.mse_loss(x_recon, x_imu[:, 1:, :])

        # --- 总 Loss ---
        # lambda 通常取 0.1 或 1.0，取决于量级
        lambda_recon = 1.0
        loss_total = loss_ce + lambda_recon * loss_mse

        # 返回 total loss
        return loss_total, None, None
    
    '''



    def save_wrapper(self, path):
        torch.save({
            'projector': self.projector.state_dict(),
            'output_head': self.output_head.state_dict()
        }, path)
        print(f"Adapters saved to {path}")

    def load_wrapper(self, path, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        sd = {k: v for k, v in sd.items() if "vq_net" not in k and "teacher" not in k}
        self.load_state_dict(sd, strict=False)
        print("Loaded Student weights.")


def get_configs():
    import random
    import numpy as np
    import argparse
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # datasets loader
    parser.add_argument('--datasets', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/ETT/', help='root path of the datasets file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='datasets file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output datasets', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='datasets loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input datasets')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # SensorLLMFuy
    parser.add_argument('--ts_backbone_yaml', type=str, default='ts_backbone.yaml', help='ts_backbone_yaml')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log_dir')
    parser.add_argument('--dataset_key', type=str, default='mhealth', help='dataset_key')
    parser.add_argument('--stage', type=int, default=1, help='stage')
    parser.add_argument('--llama_name', type=str, default=r"D:\fuy\MyCode\SensorLLM\Llama-3.2-1B", help='stage')
    parser.add_argument('--two_stage', type=int, default=1, help='two_stage')
    parser.add_argument('--freeze_llm', type=int, default=1, help='freeze_llm')
    parser.add_argument('--trainable_modules', type=str, default="sensor_proj,channel_id,recon_head,cls_head",
                        help='trainable_modules')
    parser.add_argument('--run_id', type=str, default="202501220_0956", help='trainable_modules')
    parser.add_argument('--debug_fake_llm', type=bool, default=False, help='debug_fake_llm')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    configs = get_configs()
    configs.ts_backbone_yaml = r"D:\fuy\MyCode\SensorLLMLib_v2\configs\ts_backbone.yaml"
    configs.debug_fake_llm = True

    configs.stage=2
    if torch.cuda.is_available() and configs.use_gpu:
        configs.device = torch.device('cuda:{}'.format(configs.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            configs.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            configs.device = torch.device("cpu")
        print('Using cpu or mps')
    model = AlignmentModel(configs).to("cuda:0")
    x = torch.rand(32,96,15)
    c = model(x.to("cuda:0"),None,None)
    d = 'end'