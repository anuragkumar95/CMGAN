from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
import argparse
import wandb
import psutil

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--exp", type=str, help='Experiment name')
parser.add_argument("--win_len", type=int, default=24)
parser.add_argument("--samples", type=int, default=24)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='dir to VCTK-DEMAND dataset',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


wandb.login()

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, train_ds, test_ds, win_len, samples, gpu_id: int = None):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.win_len=win_len
        self.samples = samples
        self.model = TSCNet(num_channel=16, 
                            num_features=self.n_fft // 2 + 1, 
                            win_len=self.win_len+1, 
                            gpu_id=gpu_id)
        #summary(
        #    self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        #)
        self.discriminator = discriminator.Discriminator(ndf=16)
        #summary(
        #    self.discriminator,
        #    [
        #        (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
        #        (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
        #    ],
        #)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )
        if gpu_id is not None:
            self.model = self.model.to(gpu_id)
            self.discriminator = self.discriminator.to(gpu_id)
            self.model = DDP(self.model, device_ids=[gpu_id])
            self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    
    def create_spectrograms(self, noisy, clean):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )
        #NOTE: PLEASE REMOVE COMMENTS WHEN TRAINING ON GPU
        win = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            win = win.to(self.gpu_id)

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=win,
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=win,
            onesided=True,
        )
        return noisy_spec, clean_spec
        
    def forward_generator_step(self, clean_spec, noisy_spec):
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft),#.to(self.gpu_id),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }
    
    def forward_generator_step2(self, noisy_stack):#, clean_real_stack, clean_imag_stack, clean):
        steps, b, _, _, f = noisy_stack.shape
        samples = self.samples
        for idx in range(steps):
            mini_batch = noisy_stack[idx, :, :, :, :]
            
            est_real, est_imag = self.model(mini_batch, k=samples)
            est_real, est_imag = est_real.permute(0, 1, 2, 4, 3), est_imag.permute(0, 1, 2, 4, 3)
            est_mag = torch.sqrt(est_real**2 + est_imag**2)
            #clean_mag = torch.sqrt(clean_real_stack[idx, :, :, :, (win_len//2) + 1]**2 + clean_imag_stack[idx, :, :, :, (win_len//2) + 1]**2).unsqueeze(-1)

            est_audios = []
            for k in range(samples):
                est_spec_uncompress = power_uncompress(est_real[k, ...], est_imag[k, ...]).squeeze(1)
                #Pad the est_spec on both sides since istft won't work on single frame
                pad = torch.zeros(b,1,f,2)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                est_spec_uncompress = torch.cat([pad, est_spec_uncompress, pad], dim=1)
                
                win = torch.hamming_window(self.n_fft)
                if self.gpu_id is not None:
                    win = win.to(self.gpu_id)

                est_audio = torch.istft(
                    est_spec_uncompress.permute(0,2,1,3),
                    self.n_fft,
                    self.hop,
                    window=win,
                    onesided=True,
                )
                est_audios.append(est_audio)
            
            est_audios = torch.stack(est_audios, dim=0)
            #clip out audio with the sametime index as in the current frame
            #st = idx * self.hop
            #en = st + self.n_fft
            #Account for zero pading
            #aud_pad = torch.zeros(b, self.n_fft)
            #if self.gpu_id is not None:
            #    aud_pad = aud_pad.to(self.gpu_id)
            #clean_aud = torch.cat([aud_pad, clean[:, st:en], aud_pad], dim=-1)
            
            #clean_real_slc = clean_real_stack[idx, :, :, :, (win_len//2)+1].unsqueeze(-1)
            #clean_imag_slc = clean_imag_stack[idx, :, :, :, (win_len//2)+1].unsqueeze(-1)

            yield  {"est_real": est_real,
                    "est_imag": est_imag,
                    "est_mag": est_mag,
                    #"clean_real": clean_real_slc,
                    #"clean_imag": clean_imag_slc,
                    #"clean_mag": clean_mag,
                    "est_audio": est_audios,
                    #"clean": clean_aud
                    }
    

    def calculate_generator_loss(self, generator_outputs):
       
        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"].permute(0,1,3,2), generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"].permute(0, 1, 3, 2)
        )

        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"].permute(0, 1, 3, 2)
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"].permute(0, 1, 3, 2))

        est_audio_len = generator_outputs["est_audio"].shape[-1]
        
        time_loss = torch.mean(
                torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"][:, :est_audio_len])
            )
        
        loss = (args.loss_weights[0] * loss_ri
                + args.loss_weights[1] * loss_mag
                + args.loss_weights[3] * gen_loss_GAN
                + args.loss_weights[2] * time_loss
        )
    
        return loss
    
    def calculate_generator_loss2(self, generator_outputs, samples):
        loss = 0
        for k in range(samples):
            predict_fake_metric = self.discriminator(
                generator_outputs["clean_mag"].permute(0,1,3,2), generator_outputs["est_mag"][k, ...]
            )
            gen_loss_GAN = F.mse_loss(
                predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
            )

            loss_mag = F.mse_loss(
                generator_outputs["est_mag"][k, ...], generator_outputs["clean_mag"].permute(0, 1, 3, 2)
            )

            loss_ri = F.mse_loss(
                generator_outputs["est_real"][k, ...], generator_outputs["clean_real"].permute(0, 1, 3, 2)
            ) + F.mse_loss(generator_outputs["est_imag"][k, ...], generator_outputs["clean_imag"].permute(0, 1, 3, 2))

            est_audio_len = generator_outputs["est_audio"].shape[-1]
            
            time_loss = torch.mean(
                    torch.abs(generator_outputs["est_audio"][k, :] - generator_outputs["clean"][:, :est_audio_len])
                )
            loss += (args.loss_weights[0] * loss_ri
                    + args.loss_weights[1] * loss_mag
                    + args.loss_weights[3] * gen_loss_GAN
                    + args.loss_weights[2] * time_loss
            )       
        return loss/samples

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        print("Audio:", generator_outputs['clean'].shape, generator_outputs["est_audio"].shape)
        pesq_mask, pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        print(f"PESQ:{pesq_score}, PESQ MASK:{pesq_mask}")

        if self.gpu_id is not None:
            pesq_score = pesq_score.to(self.gpu_id)
            pesq_mask = pesq_mask.to(self.gpu_id)
       
        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"].permute(0,1,3,2), generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"].permute(0,1,3,2), generator_outputs["clean_mag"].permute(0,1,3,2)
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten()*pesq_mask, pesq_score*pesq_mask)
        else:
            discrim_loss_metric = None

        return discrim_loss_metric, pesq_score
    
    def train_step2(self, batch):
        # Trainer generator
        clean = batch[0]
        noisy = batch[1]
        one_labels = torch.ones(args.batch_size)
        if self.gpu_id is not None:
            clean = batch[0].to(self.gpu_id)
            noisy = batch[1].to(self.gpu_id)
            one_labels = one_labels.to(self.gpu_id)

        win_len=self.win_len
        noisy_spec, clean_spec = self.create_spectrograms(noisy, clean)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        batch, ch, time, freq = noisy_spec.shape

        noisy_win_stack = torch.zeros(1, batch, ch, win_len+1, freq)
        clean_win_real_stack = torch.zeros(1, batch, 1, freq, win_len+1)
        clean_win_imag_stack = torch.zeros(1, batch, 1, freq, win_len+1)
        if self.gpu_id is not None:
            noisy_win_stack = noisy_win_stack.to(self.gpu_id)
            clean_win_real_stack = clean_win_real_stack.to(self.gpu_id)
            clean_win_imag_stack = clean_win_imag_stack.to(self.gpu_id)
        
        for i in range(time):
            if i < win_len//2:
                pad_len = (win_len//2) - i
                pad = torch.zeros(batch, ch, pad_len, freq)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                n_frame = noisy_spec[:,:, :i + (win_len//2)+1,:]
                #c_frame = clean_spec[:,:, :, :i + (win_len//2)+1]
                n_frame = torch.cat([pad, n_frame], dim=2)
                #pad = pad.permute(0, 1, 3, 2)
                #c_frame = torch.cat([pad, c_frame], dim=-1)
                
            elif i > time - 1 - win_len:
                pad_len = win_len - (time - 1 - i)
                pad = torch.zeros(batch, ch, pad_len, freq)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                n_frame = noisy_spec[:, :, i:, :]
                #c_frame = clean_spec[:, :, :, i:]
                n_frame = torch.cat([n_frame, pad], dim=2)
                #pad = pad.permute(0, 1, 3, 2)
                #c_frame = torch.cat([c_frame, pad], dim=-1)
                  
            else:
                n_frame = noisy_spec[:, :, i:i + win_len+1, :]
                #c_frame = clean_spec[:, :, :, i:i + win_len+1]
            
            noisy_win_stack = torch.cat([noisy_win_stack, n_frame.unsqueeze(0)], dim=0)
            #clean_win_real_stack = torch.cat([clean_win_real_stack, c_frame[:, 0, :, :].unsqueeze(1).unsqueeze(0)], dim=0)
            #clean_win_imag_stack = torch.cat([clean_win_imag_stack, c_frame[:, 1, :, :].unsqueeze(1).unsqueeze(0)], dim=0)

        #Perhaps append to a list and concatenate once will be more optimal?
        generator_outputs = {
            "est_real": [],
            "est_imag": [],
            "est_mag": [],
            "clean_real": [],
            "clean_imag": [],
            "clean_mag": [],
            "est_audio": []
        }

        outputs = {
            "est_real": [],
            "est_imag": [],
            "est_mag": [],
            "clean_real": [],
            "clean_imag": [],
            "clean_mag": [],
            "est_audio": []
        }

        loss_step=20

        #Calculate generator loss over window frames and collect outputs. 
        for i, output in enumerate(self.forward_generator_step2(noisy_win_stack[1:, :, :, :, :])): 
                                                #clean_win_real_stack[1:, :, :, :, :], 
                                                #clean_win_imag_stack[1:, :, :, :, :],
                                                #clean)):
           
            output["one_labels"] = torch.ones(args.batch_size)
            if self.gpu_id is not None:
                output["one_labels"] =  output["one_labels"].to(self.gpu_id)
            
            #Store the generator outputs and pass it to the discriminator
            outputs['est_real'].append(output['est_real'])
            outputs['est_imag'].append(output['est_imag'])
            outputs['est_mag'].append(output['est_mag'])
            outputs['est_audio'].append(output['est_audio'][..., self.n_fft:-self.n_fft])

            if (i+1) % loss_step == 1:
                st = (i+1) // loss_step
                en = min(st+20, clean_spec.shape[3])

                outputs['est_real'] = torch.stack(outputs['est_real'], dim=3).squeeze(4)
                outputs['est_imag'] = torch.stack(outputs['est_imag'], dim=2).squeeze(4)
                outputs['est_mag'] = torch.stack(outputs['est_mag'], dim=3).squeeze(4)
                outputs['clean_real'] = clean_spec[:, 0, st:en, :].unsqueeze(1)
                outputs['clean_imag'] = clean_spec[:, 1, st:en, :].unsqueeze(1)
                outputs['clean_mag'] = torch.sqrt(clean_spec[:, 0, st:en, :]**2 + clean_spec[:, 1, st:en, :]**2).unsqueeze(1)
                outputs['est_audio'] = torch.stack(outputs['est_audio'], dim=-1)
                outputs['clean'] = clean[:, st * self.hop: (st * self.hop) + (loss_step * self.n_fft)]


                loss = self.calculate_generator_loss2(outputs, samples=self.samples)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Generator loop step:{i}, Loss:{loss}") 

                outputs = {
                            "est_real": [],
                            "est_imag": [],
                            "est_mag": [],
                            "clean_real": [],
                            "clean_imag": [],
                            "clean_mag": [],
                            "est_audio": []
                        }
            
            generator_outputs['est_real'].append(output['est_real'].mean(0))
            generator_outputs['est_imag'].append(output['est_imag'].mean(0))
            generator_outputs['est_mag'].append(output['est_mag'].mean(0))
            #generator_outputs['est_audio'].append(output['est_audio'].mean(0)[..., self.n_fft:-self.n_fft])
            
        generator_outputs['est_real'] = torch.stack(generator_outputs['est_real'], dim=2).squeeze(3)
        generator_outputs['est_imag'] = torch.stack(generator_outputs['est_imag'], dim=2).squeeze(3)
        generator_outputs['est_mag'] = torch.stack(generator_outputs['est_mag'], dim=2).squeeze(3)
        generator_outputs['clean_real'] = clean_spec[:, 0, :, :].unsqueeze(1)
        generator_outputs['clean_imag'] = clean_spec[:, 1, :, :].unsqueeze(1)
        generator_outputs['clean_mag'] = torch.sqrt(clean_spec[:, 0, :, :]**2 + clean_spec[:, 1, :, :]**2).unsqueeze(1)
        
        est_spec_uncompress = power_uncompress(generator_outputs['est_real'], 
                                               generator_outputs['est_imag']).squeeze(1)
        
        win = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            win = win.to(self.gpu_id)
        
        est_audio = torch.istft(
                        est_spec_uncompress.permute(0,2,1,3),
                        self.n_fft,
                        self.hop,
                        window=win,
                        onesided=True,
                    )
        generator_outputs['est_audio'] = est_audio
        generator_outputs['clean'] = clean
        generator_outputs['one_labels'] = one_labels

        generator_loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        generator_loss.backward()
        self.optimizer.step()

        print(f"Generator_loss:{loss}")

        # Train Discriminator
        discrim_loss_metric, pesq_score = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

    
        yield loss.item(), discrim_loss_metric.item(), pesq_score.mean().item()


    def train_step(self, batch):
        # Trainer generator
        clean = batch[0]
        noisy = batch[1]
        one_labels = torch.ones(args.batch_size)
        if self.gpu_id:
            clean = batch[0].to(self.gpu_id)
            noisy = batch[1].to(self.gpu_id)
            one_labels = torch.ones(args.batch_size).to(self.gpu_id)
        
        noisy_spec, clean_spec = self.create_spectrograms(noisy, clean)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)

        generator_outputs = self.forward_generator_step(
            clean_spec,
            noisy_spec,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step2(self, batch):
        # Trainer generator
        clean = batch[0]
        noisy = batch[1]
        one_labels = torch.ones(args.batch_size)
        if self.gpu_id is not None:
            clean = batch[0].to(self.gpu_id)
            noisy = batch[1].to(self.gpu_id)
            one_labels = one_labels.to(self.gpu_id)

        win_len=self.win_len
        noisy_spec, clean_spec = self.create_spectrograms(noisy, clean)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        batch, ch, time, freq = noisy_spec.shape

        noisy_win_stack = torch.zeros(1, batch, ch, win_len+1, freq)
        clean_win_real_stack = torch.zeros(1, batch, 1, freq, win_len+1)
        clean_win_imag_stack = torch.zeros(1, batch, 1, freq, win_len+1)
        if self.gpu_id is not None:
            noisy_win_stack = noisy_win_stack.to(self.gpu_id)
            clean_win_real_stack = clean_win_real_stack.to(self.gpu_id)
            clean_win_imag_stack = clean_win_imag_stack.to(self.gpu_id)

        for i in range(time):
            if i < win_len//2:
                pad_len = (win_len//2) - i
                pad = torch.zeros(batch, ch, pad_len, freq)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                n_frame = noisy_spec[:,:, :i + (win_len//2)+1,:]
                c_frame = clean_spec[:,:, :, :i + (win_len//2)+1]
                n_frame = torch.cat([pad, n_frame], dim=2)
                pad = pad.permute(0, 1, 3, 2)
                c_frame = torch.cat([pad, c_frame], dim=-1)
                
            elif i > time - 1 - win_len:
                pad_len = win_len - (time - 1 - i)
                pad = torch.zeros(batch, ch, pad_len, freq)
                if self.gpu_id is not None:
                    pad = pad.to(self.gpu_id)
                n_frame = noisy_spec[:, :, i:, :]
                c_frame = clean_spec[:, :, :, i:]
                n_frame = torch.cat([n_frame, pad], dim=2)
                pad = pad.permute(0, 1, 3, 2)
                c_frame = torch.cat([c_frame, pad], dim=-1)
                  
            else:
                n_frame = noisy_spec[:, :, i:i + win_len+1, :]
                c_frame = clean_spec[:, :, :, i:i + win_len+1]
            
            noisy_win_stack = torch.cat([noisy_win_stack, n_frame.unsqueeze(0)], dim=0)
            clean_win_real_stack = torch.cat([clean_win_real_stack, c_frame[:, 0, :, :].unsqueeze(1).unsqueeze(0)], dim=0)
            clean_win_imag_stack = torch.cat([clean_win_imag_stack, c_frame[:, 1, :, :].unsqueeze(1).unsqueeze(0)], dim=0)

        generator_outputs = {
            "est_real": [],
            "est_imag": [],
            "est_mag": [],
            "clean_real": [],
            "clean_imag": [],
            "clean_mag": [],
            "est_audio": []
        }
    
        #Calculate generator loss over window frames and collect outputs. 
        for i, outputs in enumerate(self.forward_generator_step2(noisy_win_stack[1:, :, :, :, :], 
                                                   clean_win_real_stack[1:, :, :, :, :], 
                                                   clean_win_imag_stack[1:, :, :, :, :],
                                                   clean)):
            outputs["one_labels"] = torch.ones(args.batch_size)
            if self.gpu_id is not None:
                outputs["one_labels"] =  outputs["one_labels"].to(self.gpu_id)

            #Store the generator outputs and pass it to the discriminator
            generator_outputs['est_real'].append(outputs['est_real'])
            generator_outputs['est_imag'].append(outputs['est_imag'])
            generator_outputs['est_mag'].append(outputs['est_mag'])
           
        generator_outputs['est_real'] = torch.stack(generator_outputs['est_real'], dim=3).squeeze(4)
        generator_outputs['est_imag'] = torch.stack(generator_outputs['est_imag'], dim=3).squeeze(4)
        generator_outputs['est_mag'] = torch.stack(generator_outputs['est_mag'], dim=3).squeeze(4)
        generator_outputs['clean_real'] = clean_spec[:, 0, :, :].unsqueeze(1)
        generator_outputs['clean_imag'] = clean_spec[:, 1, :, :].unsqueeze(1)
        generator_outputs['clean_mag'] = torch.sqrt(clean_spec[:, 0, :, :]**2 + clean_spec[:, 1, :, :]**2).unsqueeze(1)
       
        
        est_spec_uncompress = power_uncompress(generator_outputs['est_real'], 
                                               generator_outputs['est_imag']).squeeze(1).squeeze(2)
    
        win = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            win = win.to(self.gpu_id)
        
        est_auds = []
        for spec in est_spec_uncompress:
            est_audio = torch.istft(
                        spec.permute(0,2,1,3),
                        self.n_fft,
                        self.hop,
                        window=win,
                        onesided=True,
                    )
            est_auds.append(est_audio)
        est_auds = torch.stack(est_auds, dim=0)
        
        generator_outputs['est_audio'] = est_audio
        generator_outputs['clean'] = clean
        generator_outputs['one_labels'] = one_labels
 
        loss = self.calculate_generator_loss2(generator_outputs)

        discrim_loss_metric, pesq_score = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item(), pesq_score.mean().item()


    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    def test2(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            for loss, disc_loss, pesq in self.test_step2(batch):
                gen_loss_total += loss
                disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        wandb.log({"Test_D_loss": disc_loss_avg,
                   "Test_G_loss": gen_loss_avg,
                   "Test_PESQ": pesq,
                   "Epoch": step})
        print(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train2(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        wandb.init(project=args.exp)
        for epoch in range(args.epochs):
            self.model.train()
            self.discriminator.train()
            step = 0
            for _, batch in enumerate(self.train_ds):
                for loss, disc_loss, pesq in self.train_step2(batch):
                    template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                    wandb.log({"Train_D_loss": disc_loss,
                               "Train_G_loss": loss,
                               "Train_PESQ": pesq,
                               "Train Step": step})
                    print(template.format(self.gpu_id, epoch, step, loss, disc_loss))
                    if (step % args.log_interval) == 0:
                        logging.info(
                            template.format(self.gpu_id, epoch, step, loss, disc_loss)
                        )
                    step += 1
                scheduler_G.step()
                scheduler_D.step()
            gen_loss = self.test2()
            path = os.path.join(
                args.save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if self.gpu_id == 0:
                torch.save(self.model.module.state_dict(), path)

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        for epoch in range(args.epochs):
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % args.log_interval) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss, disc_loss)
                    )
            gen_loss = self.test2()
            path = os.path.join(
                args.save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if self.gpu_id == 0:
                torch.save(self.model.module.state_dict(), path)
            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int, args):
    
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(f"Available gpus:{available_gpus}")
    #print("AAAA")
    
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 1, args.cut_len
    )
    #print(f"Train:{len(train_ds)}, Test:{len(test_ds)}")
    trainer = Trainer(train_ds, test_ds, args.win_len, args.samples, rank)
    trainer.train2()
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
    #main(None, world_size, args)