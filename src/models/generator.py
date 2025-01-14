from models.conformer import ConformerBlock
import torch
import torch.nn as nn


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64, out_channel=2):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1, signal_window=51, gpu_id=None):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)
        self.out_mu = nn.Linear(signal_window, 1)
        self.out_var = nn.Linear(signal_window, 1)
        self.N = torch.distributions.Normal(0, 1)
        self.gpu_id = gpu_id

    def sample(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        x = mu + sigma * self.N.sample(mu.shape).to(self.gpu_id)
        x = self.prelu_out(x)
        return x.permute(0, 2, 1).unsqueeze(1)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        x_mu = self.out_mu(x)
        x_var = self.out_var(x)
        return x_mu, x_var

class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64, signal_window=51, gpu_id=None):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))
        self.out_mu = nn.Linear(signal_window, 1)
        self.out_var = nn.Linear(signal_window, 1)
        self.N = torch.distributions.Normal(0, 1)
        self.gpu_id = gpu_id

    def sample(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        x = mu + sigma * self.N.sample(mu.shape).to(self.gpu_id)
        return x

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        x_mu = self.out_mu(x.permute(0,1,3,2))
        x_var = self.out_var(x.permute(0,1,3,2))
        return x_mu, x_var

class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201, win_len=51, mag_only=False, gpu_id=None):
        super(TSCNet, self).__init__()
        if mag_only:
            in_channels=1
        else:
            in_channels=3
        self.dense_encoder = DenseEncoder(in_channel=in_channels, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1, signal_window=win_len, gpu_id=gpu_id
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel, signal_window=win_len, gpu_id=gpu_id)

    def forward(self, x, mag_only=False):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
    
        win_len = mag.shape[2]
        
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        if mag_only:
            x_in = mag
        else:
            x_in = torch.cat([mag, x], dim=1)
       
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        #Sample mask from the output distribution k times and take the average.
        mask_mu, mask_var = self.mask_decoder(out_5)
        mask = self.mask_decoder.sample(mask_mu, mask_var)
        
        #Output mask is for the middle frame of the window
        out_mag = mask * mag[:, :, win_len//2 + 1, :].unsqueeze(2)
        mag_real = (out_mag * torch.cos(noisy_phase[:, :, win_len//2 + 1, :].unsqueeze(2))).permute(0, 1, 3, 2)
        mag_imag = (out_mag * torch.sin(noisy_phase[:, :, win_len//2 + 1, :].unsqueeze(2))).permute(0, 1, 3, 2)
        
        if not mag_only:
            complex_mu, complex_var = self.complex_decoder(out_5)
            complex_out = self.complex_decoder.sample(complex_mu, complex_var)
            final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
            final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
            return final_real, final_imag
        
        return mag_real, mag_imag
            
