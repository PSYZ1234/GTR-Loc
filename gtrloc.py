import re
import torch
import logging
import torch.nn as nn
import torch_scatter
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch.nn.functional as F
from collections import OrderedDict
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.pose_util import zone_ids_to_positions, zone_ids_to_directions
from utils.pointnet_util import PointNetSetAbstraction

_tokenizer = _Tokenizer()
_logger = logging.getLogger(__name__)


class PosPromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        n_ctx = 3
        ctx_init = "positioned in District"
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.dtype = dtype
        self.clip_model = clip_model
        self.meta_net = TransformerDecoderLayer(512, 8, 0.1)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (B, 1, D)
                ctx,     # (B, n_ctx, D)
                suffix,  # (B, *, D)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(im_features.device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        prefix = embedding[:, :1, :].detach()
        suffix = embedding[:, 1 + self.n_ctx :, :].detach()
        cti = self.ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1) 
        ctx = self.meta_net(cti, im_features.unsqueeze(1))  # (B, n_ctx, D)
        prompts = self.construct_prompts(ctx, prefix, suffix)  # (B, T, D)

        return prompts + embedding, tokenized_prompts
    

class OriPromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        n_ctx = 3
        ctx_init = "oriented toward the"
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.dtype = dtype
        self.clip_model = clip_model
        self.meta_net = TransformerDecoderLayer(512, 8, 0.1)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (B, 1, D)
                ctx,     # (B, n_ctx, D)
                suffix,  # (B, *, D)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(im_features.device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        prefix = embedding[:, :1, :].detach()
        suffix = embedding[:, 1 + self.n_ctx :, :].detach()
        cti = self.ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1) 
        ctx = self.meta_net(cti, im_features.unsqueeze(1))  # (B, n_ctx, D)
        prompts = self.construct_prompts(ctx, prefix, suffix)  # (B, T, D)

        return prompts + embedding, tokenized_prompts
    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

def Norm(norm_type, num_feats, bn_momentum=0.1, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class Conv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 dimension=3):
        super(Conv, self).__init__()

        self.net = nn.Sequential(ME.MinkowskiConvolution(inplanes,
                                                         planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         dilation=dilation,
                                                         bias=bias,
                                                         dimension=dimension),)

    def forward(self, x):
        return self.net(x)


class Encoder(ME.MinkowskiNetwork):
    """
    FCN encoder, used to extract features from the input point clouds.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels, norm_type, D=3):
        super(Encoder, self).__init__(D)

        self.in_channels = 3
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.conv_planes = [32, 64, 128, 256, 256, 256, 256, 512, 512]

        # in_channels, conv_planes, kernel_size, stride  dilation  bias
        self.conv1 = Conv(self.in_channels, self.conv_planes[0], 3, 1, 1, True)
        self.conv2 = Conv(self.conv_planes[0], self.conv_planes[1], 3, 2, bias=True)
        self.conv3 = Conv(self.conv_planes[1], self.conv_planes[2], 3, 2, bias=True)
        self.conv4 = Conv(self.conv_planes[2], self.conv_planes[3], 3, 2, bias=True)

        self.res1_conv1 = Conv(self.conv_planes[3], self.conv_planes[4], 3, 1, bias=True)
        # 1
        self.res1_conv2 = Conv(self.conv_planes[4], self.conv_planes[5], 1, 1, bias=True)
        self.res1_conv3 = Conv(self.conv_planes[5], self.conv_planes[6], 3, 1, bias=True)

        self.res2_conv1 = Conv(self.conv_planes[6], self.conv_planes[7], 3, 1, bias=True)
        # 2
        self.res2_conv2 = Conv(self.conv_planes[7], self.conv_planes[8], 1, 1, bias=True)
        self.res2_conv3 = Conv(self.conv_planes[8], self.out_channels, 3, 1, bias=True)

        self.res2_skip = Conv(self.conv_planes[6], self.out_channels, 1, 1, bias=True)

    def forward(self, x):
        """
        w/o BN
        """

        x = MEF.relu(self.conv1(x))
        x = MEF.relu(self.conv2(x))
        x = MEF.relu(self.conv3(x))
        res = MEF.relu(self.conv4(x))

        x = MEF.relu(self.res1_conv1(res))
        x = MEF.relu(self.res1_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res1_conv3(x))

        res = res + x

        x = MEF.relu(self.res2_conv1(res))
        x = MEF.relu(self.res2_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


def one_hot(x, N):
    one_hot = torch.FloatTensor(x.size(0), N, x.size(1), x.size(2)).zero_().to(x.device)
    one_hot = one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


class CondLayer(nn.Module):
    """
    pixel-wise feature modulation.
    """
    def __init__(self, in_channels):
        super(CondLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x, gammas, betas):
        return F.relu(self.bn((gammas * x) + betas))
        # return F.relu((gammas * x) + betas)


class Cls_Head(nn.Module):
    """
    Classification Head
    """
    def __init__(self, in_channels=512, level_cluster=25):
        super(Cls_Head, self).__init__()

        channels_c = [512, 256, level_cluster]

        # level network.
        self.conv1_l1 = nn.Linear(in_channels, channels_c[0])
        self.norm1_l1 = nn.BatchNorm1d(channels_c[0])
        self.conv2_l1 = nn.Linear(channels_c[0], channels_c[1])
        self.norm2_l1 = nn.BatchNorm1d(channels_c[1])
        self.conv3_l1 = nn.Linear(channels_c[1], channels_c[2])
        self.dp1_l1 = nn.Dropout(0.5)
        self.dp2_l1 = nn.Dropout(0.5)


    def forward(self, res):

        x1 = self.dp1_l1(F.relu(self.norm1_l1(self.conv1_l1(res))))
        x1 = self.dp2_l1(F.relu(self.norm2_l1(self.conv2_l1(x1))))
        # output the classification probability.
        out_lbl_1 = self.conv3_l1(x1)

        return out_lbl_1


class PTEncoder(nn.Module):
    def __init__(self):
        super(PTEncoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(512,  4,    32,   3,       [32, 32, 64],     False, False)
        self.sa2 = PointNetSetAbstraction(128,  8,    16,   64 + 3,  [64, 128, 256],   False, False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], False, True)

    def forward(self, xyz):
        B                 = xyz.size(0)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points         = l3_points.view(B, -1) 

        return l3_points.max(dim=1)[0]  # [B, 512]
    

class Text_Head(nn.Module):
    def __init__(self, feature_dim, norm_type, in_channel, mlp, num_pos=100, num_ori=16):
        super(Text_Head, self).__init__()
        self.ptencoder = Encoder(out_channels=feature_dim, norm_type=norm_type)
        self.mlp_fc_pos = nn.ModuleList()
        self.mlp_bn_pos = nn.ModuleList()
        self.mlp_fc_ori = nn.ModuleList()
        self.mlp_bn_ori = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fc_pos.append(nn.Linear(last_channel, out_channel))
            self.mlp_bn_pos.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fc_ori.append(nn.Linear(last_channel, out_channel))
            self.mlp_bn_ori.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.fc_pos_finall = nn.Linear(mlp[-1], num_pos)
        self.fc_ori_finall = nn.Linear(mlp[-1], num_ori)


    def forward(self, input):
        features = self.ptencoder(input)
        globalx = torch_scatter.scatter_max(features.F, features.C[:, 0].long(), dim=0)[0]  # [256, 512]    
        t = globalx
        q = globalx

        for i, fc_pos in enumerate(self.mlp_fc_pos):
            bn_pos = self.mlp_bn_pos[i]
            t      = self.dropout(F.relu(bn_pos(fc_pos(t))))  # [B, D]

        for j, fc_ori in enumerate(self.mlp_fc_ori):
            bn_ori = self.mlp_bn_ori[j]
            q      = self.dropout(F.relu(bn_ori(fc_ori(q))))  # [B, D]

        pos_cls = self.fc_pos_finall(t)  
        ori_cls = self.fc_ori_finall(t)  
        pos_cls  = F.log_softmax(pos_cls, dim=1)  # [B, D]
        ori_cls  = F.log_softmax(ori_cls, dim=1)  # [B, D]
        
        return pos_cls, ori_cls
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class PoseRegressor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PoseRegressor, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.fct = nn.Linear(mlp[-1], 3)
        self.fcq = nn.Linear(mlp[-1], 3)

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(fc(x)))  # [B, D]

        t = self.fct(x)  
        q = self.fcq(x)   
        
        return t, q
    

class Reg_Head_Stu(nn.Module):
    def __init__(self, num_head_blocks, in_channels=512, mlp_ratio=1.0):
        super(Reg_Head_Stu, self).__init__()
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = in_channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels \
            else nn.Linear(self.in_channels, self.head_channels)

        block_channels = int(self.head_channels * mlp_ratio)
        self.res3_conv1 = nn.Linear(self.in_channels, self.head_channels)
        self.res3_conv2 = nn.Linear(self.head_channels, block_channels)
        self.res3_conv3 = nn.Linear(block_channels, self.head_channels)

        self.res_blocks = []
        self.norm_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Linear(self.head_channels, self.head_channels),
                nn.Linear(self.head_channels, block_channels),
                nn.Linear(block_channels, self.head_channels),
            ))

            super(Reg_Head_Stu, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Reg_Head_Stu, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Reg_Head_Stu, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Linear(self.head_channels, self.head_channels)
        self.fc2 = nn.Linear(self.head_channels, block_channels)
        self.fc3 = nn.Linear(block_channels, 3)

    def forward(self, res):
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:

            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        return sc
    

class Reg_Head(nn.Module):
    """
    nn.Linear版
    """
    def __init__(self, clip_model, num_head_blocks, in_channels=512, mlp_ratio=1.0):
        super(Reg_Head, self).__init__()
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = in_channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels \
            else nn.Linear(self.in_channels, self.head_channels)

        block_channels = int(self.head_channels * mlp_ratio)
        self.res3_conv1 = nn.Linear(self.in_channels, self.head_channels)
        self.res3_conv2 = nn.Linear(self.head_channels, block_channels)
        self.res3_conv3 = nn.Linear(block_channels, self.head_channels)

        self.res_blocks = []
        self.norm_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Linear(self.head_channels, self.head_channels),
                nn.Linear(self.head_channels, block_channels),
                nn.Linear(block_channels, self.head_channels),
            ))

            super(Reg_Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Reg_Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Reg_Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Linear(self.head_channels, self.head_channels)
        self.fc2 = nn.Linear(self.head_channels, block_channels)
        self.fc3 = nn.Linear(block_channels, 3)

        # text
        self.pos_prompt_learner = PosPromptLearner(clip_model)
        self.ori_prompt_learner = OriPromptLearner(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.dtype = next(clip_model.parameters()).dtype
        # self.fctext = nn.Linear(512, 512+25)
        # self.fcvisual = nn.Linear(512+25, 512)
        self.fctext = nn.Linear(512, 512+100)
        self.fcvisual = nn.Linear(512+100, 512)
        self.transformer_decoder = TransformerDecoderLayer(512, 8, 0.1)

        # stu learner
        self.Reg_Head_Stu = Reg_Head_Stu(num_head_blocks, in_channels, mlp_ratio)
        self.learner1 = nn.Linear(self.in_channels, self.head_channels)
        self.learner2 = nn.Linear(self.head_channels, block_channels)
        self.learner3 = nn.Linear(block_channels, self.head_channels)

    def forward(self, res, cls_pos, cls_ori, idx):
        # stu learning
        learner = F.relu(self.learner1(res))
        learner = F.relu(self.learner2(learner))
        learner = F.relu(self.learner3(learner))
        res_stu = res + 0.1 * F.normalize(learner, dim=-1, p=2)
        sc_stu = self.Reg_Head_Stu(res_stu)

        # maxpool for visual
        global_visual = self.fcvisual(torch_scatter.scatter_max(learner, idx, dim=0)[0])  # [256, 512]
        global_visual = F.normalize(global_visual, dim=-1, p=2)

        # text position
        classnames_zone_pos = zone_ids_to_positions(cls_pos)
        pos_prompts, pos_tokenized_prompts = self.pos_prompt_learner(global_visual, classnames_zone_pos)
        pos_text_features_list = []
        for pts_i, tts_i in zip(pos_prompts, pos_tokenized_prompts):
            with torch.no_grad():
                pos_text_features = self.text_encoder(pts_i.unsqueeze(0), tts_i.unsqueeze(0))
            pos_text_features_list.append(pos_text_features)  # (n_cls_i, D)
        pos_global_text = torch.cat(pos_text_features_list, dim=0)  # (n_cls, D)

        # text orientation
        classnames_zone_ori = zone_ids_to_directions(cls_ori)
        ori_prompts, ori_tokenized_prompts = self.ori_prompt_learner(global_visual, classnames_zone_ori)
        ori_text_features_list = []
        for pts_i, tts_i in zip(ori_prompts, ori_tokenized_prompts):
            with torch.no_grad():
                ori_text_features = self.text_encoder(pts_i.unsqueeze(0), tts_i.unsqueeze(0))
            ori_text_features_list.append(ori_text_features)  # (n_cls_i, D)
        ori_global_text = torch.cat(ori_text_features_list, dim=0)  # (n_cls, D)
        
        # text with visual
        global_text = pos_global_text + ori_global_text  # [256, 512]
        global_text = F.normalize(global_text, dim=-1, p=2)
        text_diff = self.transformer_decoder(global_text.unsqueeze(1), global_visual.unsqueeze(1))
        text_diff = text_diff.squeeze(1)
        text_visual = self.fctext(text_diff)
        text_visual = F.normalize(text_visual, dim=-1, p=2)
        res = res + 0.1 * text_visual[idx]

        # regressor
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))
        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        return sc, sc_stu


class Regressor(ME.MinkowskiNetwork):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, num_head_blocks, num_encoder_features, level_clusters=25,
                 mlp_ratio=1.0, sample_cls=False, D=3):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__(D)

        self.feature_dim = num_encoder_features
        """
        ACE
        """
        clip_model, _ = clip.load('ViT-B/32', device="cpu", jit=False)
        clip_model.float()
        clip_model.visual = None
        print("Building custom CLIP")

        # self.text_heads = Text_Head(feature_dim = self.feature_dim, norm_type='BN', in_channel=512, mlp=[512, 512, 512, 512], num_pos=100, num_ori=16)

        self.encoder = Encoder(out_channels=self.feature_dim, norm_type='BN')
        self.cls_heads = Cls_Head(in_channels=self.feature_dim, level_cluster=level_clusters)
        if not sample_cls:
            self.reg_heads = Reg_Head(clip_model=clip_model, num_head_blocks=num_head_blocks, in_channels=self.feature_dim + level_clusters,
                                      mlp_ratio=mlp_ratio)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, classifier_state_dict=None,
                            num_head_blocks=None, level_clusters=25, mlp_ratio=1.0, sample_cls=False):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        classifier_state_dict: trained classifier state dictionary
        num_head_blocks: How many extra residual blocks to use in the head.
        level_cluster: How many classification categories.
        mlp_ratio: Channel expansion ratio.
        sample_cls: training for
        """

        # Number of output channels of the last encoder layer.
        # for name, param in encoder_state_dict.items():
        #     print(f"Parameter name: {name}, Shape: {param.shape}")
        # print(encoder_state_dict.keys())
        num_encoder_features = encoder_state_dict['encoder.res2_conv3.net.0.bias'].shape[1]
        # num_encoder_features = encoder_state_dict['encoder.res2_norm3.bn.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(num_head_blocks, num_encoder_features, level_clusters, mlp_ratio, sample_cls)

        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if k.startswith('encoder.')}
        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        if classifier_state_dict!=None:
            regressor.cls_heads.load_state_dict(classifier_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^reg_heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.net.0.bias'].shape[1]
        num_decoder_features = state_dict['cls_heads.conv1_l1.weight'].shape[1]
        head_channels = state_dict['cls_heads.conv1_l1.weight'].shape[0]
        level_clusters = state_dict['cls_heads.conv3_l1.weight'].shape[0]
        reg = any(key.startswith("reg_heads") for key in state_dict)
        if reg:
            mlp_ratio = state_dict['reg_heads.res3_conv2.weight'].shape[0] / \
                        state_dict['reg_heads.res3_conv2.weight'].shape[1]
        else:
            mlp_ratio = 1

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tEncoder feature size: {num_encoder_features}"
                     f"\n\tDecoder feature size: {num_decoder_features}"
                     f"\n\tHead channels: {head_channels}"
                     f"\n\tMLP ratio: {mlp_ratio}")
        regressor = cls(num_head_blocks, num_encoder_features, mlp_ratio=mlp_ratio, level_clusters=level_clusters)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, cls_head_state_dict, reg_head_state_dict=None):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        # lw添加
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if
                              k.startswith('encoder.')}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in cls_head_state_dict.items():
            merged_state_dict[f"cls_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        if reg_head_state_dict != None:
            for k, v in reg_head_state_dict.items():
                merged_state_dict[f"reg_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        return cls.create_from_state_dict(merged_state_dict)


    @classmethod
    def create_from_split_state_dict2(cls, text_head_state_dict, encoder_state_dict, cls_head_state_dict, reg_head_state_dict=None):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in text_head_state_dict.items():
            merged_state_dict[f"text_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        # lw添加
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if
                              k.startswith('encoder.')}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in cls_head_state_dict.items():
            merged_state_dict[f"cls_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        if reg_head_state_dict != None:
            for k, v in reg_head_state_dict.items():
                merged_state_dict[f"reg_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        print("Keys in merged_state_dict:")
        for key in merged_state_dict.keys():
            print(key)

        return cls.create_from_state_dict(merged_state_dict)
    

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)
    
    def get_text_features(self, inputs):
        return self.gtg(inputs)

    def get_scene_coordinates(self, features, pos_cls, ori_cls, idx):
        out = self.reg_heads(features, pos_cls, ori_cls, idx)
        return out

    def get_scene_classification(self, features):
        out = self.cls_heads(features)
        return out
    
    def get_text_generation(self, inputs):
        pos, ori = self.text_heads(inputs)
        return pos, ori
    
    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.encoder(inputs)
        out = self.get_scene_coordinates(features.F)
        # out = self.get_scene_classification(features)
        out = ME.SparseTensor(
            features=out,
            coordinates=features.C,
        )

        return {'pred': out}