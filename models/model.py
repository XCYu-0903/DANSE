import torch
import torch.nn as nn

try:
    from models.transformer import TFCrossModalTransformerMultimodal, TFCrossModalTransformerUnimodal
except ImportError:
    from transformer import TFCrossModalTransformerMultimodal, TFCrossModalTransformerUnimodal
from utils import LearnableSigmoid2d


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, height, width = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, height, width))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, height, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, dilation, 0), value=0.0),
                nn.Conv2d(h.dense_channel * (i + 1), h.dense_channel, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for block in self.dense_block:
            x = block(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.dense_block = DenseBlock(h, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return x


class ComplexDecoder(nn.Module):
    def __init__(self, h, out_channel=2):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.complex_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.conv = nn.Conv2d(h.dense_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.complex_conv(x)
        x = self.conv(x)
        x = x.permute(0, 3, 2, 1)
        return x


class TFCTransformerBlockMultimodal(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.time_cross_transformer = TFCrossModalTransformerMultimodal(
            d_model=h.dense_channel,
            n_heads=h.n_head,
        )
        self.freq_cross_transformer = TFCrossModalTransformerMultimodal(
            d_model=h.dense_channel,
            n_heads=h.n_head,
        )

    def forward(self, x, text_embeds, text_key_padding_mask=None, text_time_attn_bias=None, text_time_token_mask=None):
        batch, channels, time, freq = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(batch * freq, time, channels)
        x = self.time_cross_transformer(
            x,
            text_embeds,
            text_key_padding_mask,
            text_time_attn_bias,
            text_time_token_mask,
        ) + x
        x = x.view(batch, freq, time, channels).permute(0, 2, 1, 3).contiguous().view(batch * time, freq, channels)
        x = self.freq_cross_transformer(x, text_embeds, text_key_padding_mask) + x
        x = x.view(batch, time, freq, channels).permute(0, 3, 1, 2)
        return x


class TFCTransformerBlockUnimodal(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.time_cross_transformer = TFCrossModalTransformerUnimodal(d_model=h.dense_channel, n_heads=h.n_head)
        self.freq_cross_transformer = TFCrossModalTransformerUnimodal(d_model=h.dense_channel, n_heads=h.n_head)

    def forward(self, x):
        batch, channels, time, freq = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(batch * freq, time, channels)
        x = self.time_cross_transformer(x) + x
        x = x.view(batch, freq, time, channels).permute(0, 2, 1, 3).contiguous().view(batch * time, freq, channels)
        x = self.freq_cross_transformer(x) + x
        x = x.view(batch, time, freq, channels).permute(0, 3, 1, 2)
        return x


class InteractionNetABSF(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.conv1_1 = nn.Conv2d(h.dense_channel, h.dense_channel, (1, 1))
        self.norm1_1 = nn.InstanceNorm2d(h.dense_channel, affine=True)
        self.act1_1 = nn.PReLU(h.dense_channel)
        self.conv1_2 = nn.Conv2d(h.dense_channel, h.dense_channel, (1, 1))
        self.norm1_2 = nn.InstanceNorm2d(h.dense_channel, affine=True)
        self.act1_2 = nn.PReLU(h.dense_channel)
        self.cat_conv1 = nn.Conv2d(h.dense_channel * 2, h.dense_channel, (1, 1))
        self.norm2_1 = nn.InstanceNorm2d(h.dense_channel, affine=True)
        self.cat_conv2 = nn.Conv2d(h.dense_channel * 2, h.dense_channel, (1, 1))
        self.norm2_2 = nn.InstanceNorm2d(h.dense_channel, affine=True)
        self.act_sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2):
        feature1_mid = self.act1_1(self.norm1_1(self.conv1_1(feature1)))
        feature2_mid = self.act1_2(self.norm1_2(self.conv1_2(feature2)))
        cat_feature = torch.cat((feature1_mid, feature2_mid), dim=1)
        feature1_mask = self.act_sigmoid(self.norm2_1(self.cat_conv1(cat_feature)))
        feature2_mask = self.act_sigmoid(self.norm2_2(self.cat_conv2(cat_feature)))
        feature1_refine = feature1 * feature1_mask
        feature2_refine = feature2 * feature2_mask
        feature1_final = feature1 + feature2_refine
        feature2_final = feature2 + feature1_refine
        return feature1_final, feature2_final


class DANSE_Multimodal(nn.Module):
    def __init__(self, h, a, device, text_embeds, num_tsblocks=4):
        super().__init__()
        try:
            from models.aed_fusion import AEDGuidance
        except ImportError:
            from aed_fusion import AEDGuidance
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.speech_dense_encoder = DenseEncoder(h, in_channel=2)
        self.noise_dense_encoder = DenseEncoder(h, in_channel=2)
        self.speech_embed_projection = nn.Sequential(
            nn.Linear(h.clap_feature_dim, h.clap_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(h.clap_feature_dim // 2, h.dense_channel),
        )
        self.noise_embed_projection = nn.Sequential(
            nn.Linear(h.clap_feature_dim, h.clap_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(h.clap_feature_dim // 2, h.dense_channel),
        )
        self.speech_TFCTransformer = nn.ModuleList([TFCTransformerBlockMultimodal(h) for _ in range(num_tsblocks)])
        self.noise_TFCTransformer = nn.ModuleList([TFCTransformerBlockMultimodal(h) for _ in range(num_tsblocks)])
        self.interaction_block = nn.ModuleList([InteractionNetABSF(h) for _ in range(num_tsblocks)])
        self.speech_mask_decoder = MaskDecoder(h, out_channel=1)
        self.speech_phase_decoder = PhaseDecoder(h, out_channel=1)
        self.noise_complex_decoder = ComplexDecoder(h, out_channel=2)
        self.text_embeds = text_embeds
        self.aed_fusion = AEDGuidance(h, a, device, text_embeds)

    def build_heatmap_text_attn_bias(
            self,
            heatmap,
            text_key_padding_mask,
            role="speech",
            gamma=None,
            eps=1e-5,
    ):
        if heatmap is None or text_key_padding_mask is None:
            return None
        token_count = text_key_padding_mask.size(1)
        heatmap = heatmap.to(dtype=torch.float32).clamp(0.0, 1.0)
        if heatmap.dim() == 2:
            _, _ = heatmap.size()
            token_visibility = heatmap.unsqueeze(-1).expand(-1, -1, token_count).clone()
        elif heatmap.dim() == 3:
            _, _, heatmap_token_count = heatmap.size()
            if heatmap_token_count != token_count:
                raise ValueError(
                    f"token-level heatmap must be [B, T, K] with K={token_count}, got {list(heatmap.shape)}"
                )
            token_visibility = heatmap.clone()
        else:
            raise ValueError(f"heatmap must be [B, T] or [B, T, K], got {list(heatmap.shape)}")

        if gamma is None:
            gamma = float(getattr(self.h, "ht_gamma", 1.0))
        token_visibility = token_visibility.masked_fill(text_key_padding_mask.unsqueeze(1), 1.0)
        if role == "noise" and token_count > 0:
            token_visibility[:, :, 0] = 1.0
        if role == "speech" and token_count > 0 and heatmap.mean() > 1.0 - eps:
            token_visibility.fill_(1.0)

        bias = gamma * torch.log(token_visibility.clamp(min=eps))
        if role == "noise" and token_count > 0:
            bias[:, :, 0] = 0.0
        if role == "speech" and token_count > 0 and heatmap.mean() > 1.0 - eps:
            bias.zero_()
        return bias.to(dtype=heatmap.dtype)

    def forward(
            self,
            noisy_amp,
            noisy_pha,
            noisy_audio_ac,
            audio_start,
            audio_end,
            valid_end,
            speech_idx=None,
            noise_idx=None,
            speech_heatmap_override=None,
            noise_heatmap_override=None,
            speech_text_time_attn_bias=None,
            noise_text_time_attn_bias=None,
            speech_text_time_token_mask=None,
            noise_text_time_token_mask=None,
            temporal_attn_mode=None,
    ):
        speech_embeds_fusion, noise_embeds_fusion, \
            speech_heatmap_final, noise_heatmap_final, \
            speech_text_embeds, noise_text_embeds, \
            speech_text_mask, noise_text_mask, \
            speech_token_heatmap, noise_token_heatmap = self.aed_fusion(
                noisy_audio_ac,
                audio_start,
                audio_end,
                valid_end,
                speech_idx,
                noise_idx,
                speech_heatmap_override,
                noise_heatmap_override,
            )

        speech_embeds_low = self.speech_embed_projection(speech_text_embeds)
        noise_embeds_low = self.noise_embed_projection(noise_text_embeds)

        x = torch.stack((noisy_amp, noisy_pha), dim=-1)
        x = x.permute(0, 3, 2, 1)
        x_speech = self.speech_dense_encoder(x)
        x_noise = self.noise_dense_encoder(x)

        if temporal_attn_mode == "aed":
            speech_text_time_attn_bias = self.build_heatmap_text_attn_bias(
                speech_token_heatmap if speech_token_heatmap is not None else speech_heatmap_final,
                speech_text_mask,
                role="speech",
            )
            noise_text_time_attn_bias = self.build_heatmap_text_attn_bias(
                noise_token_heatmap if noise_token_heatmap is not None else noise_heatmap_final,
                noise_text_mask,
                role="noise",
            )

        for i in range(self.num_tscblocks):
            x_speech = self.speech_TFCTransformer[i](
                x_speech,
                speech_embeds_low,
                speech_text_mask,
                speech_text_time_attn_bias,
                speech_text_time_token_mask,
            )
            x_noise = self.noise_TFCTransformer[i](
                x_noise,
                noise_embeds_low,
                noise_text_mask,
                noise_text_time_attn_bias,
                noise_text_time_token_mask,
            )
            x_speech, x_noise = self.interaction_block[i](x_speech, x_noise)

        amp_mask = self.speech_mask_decoder(x_speech)
        denoised_amp = noisy_amp * amp_mask
        noise_com = self.noise_complex_decoder(x_noise)
        denoised_pha = self.speech_phase_decoder(x_speech)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)),
            dim=-1,
        )
        return denoised_amp, denoised_pha, denoised_com, noise_com, speech_embeds_fusion, noise_embeds_fusion


class DANSE_Unimodal(nn.Module):
    def __init__(self, h, a=None, device=None, num_tsblocks=2):
        super().__init__()
        self.num_tscblocks = num_tsblocks
        self.speech_dense_encoder = DenseEncoder(h, in_channel=2)
        self.noise_dense_encoder = DenseEncoder(h, in_channel=2)
        self.speech_TFCTransformer = nn.ModuleList([TFCTransformerBlockUnimodal(h) for _ in range(num_tsblocks)])
        self.noise_TFCTransformer = nn.ModuleList([TFCTransformerBlockUnimodal(h) for _ in range(num_tsblocks)])
        self.interaction_block = nn.ModuleList([InteractionNetABSF(h) for _ in range(num_tsblocks)])
        self.speech_mask_decoder = MaskDecoder(h, out_channel=1)
        self.speech_phase_decoder = PhaseDecoder(h, out_channel=1)
        self.noise_complex_decoder = ComplexDecoder(h, out_channel=2)

    def forward(self, noisy_amp, noisy_pha):
        x = torch.stack((noisy_amp, noisy_pha), dim=-1)
        x = x.permute(0, 3, 2, 1)
        x_speech = self.speech_dense_encoder(x)
        x_noise = self.noise_dense_encoder(x)

        for i in range(self.num_tscblocks):
            x_speech = self.speech_TFCTransformer[i](x_speech)
            x_noise = self.noise_TFCTransformer[i](x_noise)
            x_speech, x_noise = self.interaction_block[i](x_speech, x_noise)

        amp_mask = self.speech_mask_decoder(x_speech)
        denoised_amp = noisy_amp * amp_mask
        noise_com = self.noise_complex_decoder(x_noise)
        denoised_pha = self.speech_phase_decoder(x_speech)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)),
            dim=-1,
        )
        return denoised_amp, denoised_pha, denoised_com, noise_com
