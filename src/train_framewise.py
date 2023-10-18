from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress, copy_weights, freeze_layers, original_pesq
import logging
from torchinfo import summary
import argparse
import wandb
import psutil
from speech_enh_env import SpeechEnhancementAgent

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--parallel", action='store_true', help="Set this falg to run parallel gpu training.")
parser.add_argument("--gpu", action='store_true', help="Set this falg to run single gpu training.")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--exp", type=str, default='default', help='Experiment name')
parser.add_argument("--win_len", type=int, default=24)
parser.add_argument("--samples", type=int, default=24)
parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved cmgan checkpoint for resuming training.")
parser.add_argument("--pretrain", type=str, required=False, default=None, 
                    help="path to the pretrained checkpoint of original CMGAN.")
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, required=True,
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, required=True,
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


wandb.login()

class FrameLevelTrainer:
    def __init__(self, train_ds, test_ds, win_len, samples, batchsize, parallel=False, gpu_id=None, pretrain=None, resume_pt=None):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.win_len=win_len
        self.samples = samples
        self.model = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1, 
                            win_len=self.win_len+1, 
                            gpu_id=gpu_id)
        self.batchsize=batchsize
      
        self.discriminator = discriminator.Discriminator(ndf=16)

        if pretrain is not None:
            #Load checkpoint
            print(f"Loading pretrained model saved at {args.ckpt}...")
            cmgan_state_dict = torch.load(pretrain, map_location=torch.device('cpu'))
            #Copy weights and freeze weights which are copied
            keys, self.model = copy_weights(cmgan_state_dict, self.model)
            self.model = freeze_layers(self.model, keys)
            #Free mem
            del cmgan_state_dict

        #optimizers and schedulers
        self.optimizer = torch.optim.AdamW(filter(lambda layer:layer.requires_grad,self.actor.parameters()), 
                                           lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )

        if resume_pt is not None:
            state_dict = torch.load(resume_pt, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict['generator_state_dict'])
            self.discriminator.load_state_dict(state_dict['discriminator_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_G_state_dict'])
            self.optimizer_disc.load_state_dict(state_dict['optimizer_D_state_dict'])
            self.scheduler_G.load_state_dict(state_dict['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(state_dict['scheduler_D_state_dict'])
            print(f"Loaded checkpoint saved at {resume_pt}")

        if gpu_id is not None:
            self.model = self.model.to(gpu_id)
            self.discriminator = self.discriminator.to(gpu_id)
            if parallel:
                self.model = DDP(self.model, device_ids=[gpu_id])
                self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id
        wandb.init(project=args.exp)

    
    def create_spectrograms(self, noisy, clean):
        """
        Create spectrograms from input waveform.
        ARGS:
            clean : clean waveform (batch * cut_len)
            noisy : noisy waveform (batch * cut_len)

        Return
            noisy_spec : (b * 2 * f * t) noisy spectrogram
            clean_spec : (b * 2 * f * t) clean spectrogram
            clean_real : (b * 1 * f * t) real part of clean spectrogram
            clean_imag : (b * 1 * f * t) imag part of clean spectrogram
            clean_mag  : (b * 1 * f * t) mag of clean spectrogram
        """
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

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
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        return noisy_spec, clean_spec, clean_real, clean_imag, clean_mag, clean
    
    
    def preprocess_batch(self, batch):
        """
        Converts a batch of audio waveforms and returns a batch of
        spectrograms.
        ARGS:
            batch : (b * cut_len) waveforms.

        Returns:
            Dict of spectrograms
        """
        clean, noisy, _ = batch
        one_labels = torch.ones(args.batch_size)
        if self.gpu_id is not None:
            clean = clean.to(self.gpu_id)
            noisy = noisy.to(self.gpu_id)
            one_labels = one_labels.to(self.gpu_id)

        noisy_spec, clean_spec, clean_real, clean_imag, clean_mag, cl_aud = self.create_spectrograms(clean, noisy)
        
        ret_val = {'noisy':noisy_spec,
                   'clean':clean_spec,
                   'clean_real':clean_real,
                   'clean_imag':clean_imag,
                   'clean_mag':clean_mag,
                   'cl_audio':cl_aud,
                   'one_labels':one_labels
                  }
        
        return ret_val
    
    def forward_generator_loop(self, agent):
        """
        Runs the generator for all frames and predicts mask and collects them.
        ARGS:
            noisy_spec : batch of noisy spectograms (b * ch * t * f)
            clean_spec : batch of clean spectograms (b * ch * t * f)
        """
        est_real = []
        est_imag = []
        for idx in range(agent.steps):
            #Get input
            inp = agent.get_state_input(agent.state, idx)

            #Get predictions from generator on this frame
            frame_real, frame_imag = self.model(inp)

            #Collect frames
            est_real.append(frame_real)
            est_imag.append(frame_imag)

        est_real = torch.stack(est_real, dim=2)
        est_imag = torch.stack(est_imag, dim=2)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_real = agent.state['clean_real']
        clean_imag = agent.state['clean_imag']
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
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
            "clean":agent.state['cl_audio']
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
            ) + F.mse_loss(predict_enhance_metric.flatten() * pesq_mask, pesq_score * pesq_mask)
        else:
            discrim_loss_metric = None

        pesq_score = pesq_score * pesq_mask
        return discrim_loss_metric, pesq_score.mean()

    def train_one_step(self, batch):
        agent = SpeechEnhancementAgent(window=self.win_len // 2, 
                                       n_fft=self.n_fft,
                                       hop=self.hop,
                                       gpu_id=self.gpu_id)
        agent.set_batch(batch)
        generator_outputs = self.forward_generator_loop(agent)
        
        #calculate generator_loss and update generator
        gen_loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        gen_loss.backward()
        self.optimizer.step()

        # update discriminator
        disc_loss, pesq = self.calculate_discriminator_loss(generator_outputs)
        if disc_loss is not None:
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()
        else:
            disc_loss = torch.tensor([0.0])

        pesq = original_pesq(pesq)
        return gen_loss.item(), disc_loss.item(), pesq  

    def train_one_epoch(self):
        gen_ep_loss = 0
        disc_ep_loss = 0
        ep_pesq = 0
        steps = len(self.train_ds)
        self.model.train()
        self.discriminator.train()
        for i, batch in enumerate(self.train_ds):
            batch = self.preprocess_batch(batch)
            step_gen_loss, step_disc_loss, step_pesq = self.train_one_step(batch)
            wandb.log({
                'step': i+1,
                'step_gen_loss':step_gen_loss,
                'step_disc_loss':step_disc_loss,
                'step_train_pesq':step_pesq
            })
            gen_ep_loss += step_gen_loss
            disc_ep_loss += step_disc_loss
            ep_pesq += step_pesq

        gen_ep_loss = gen_ep_loss / steps
        disc_ep_loss = disc_ep_loss / steps
        ep_pesq = ep_pesq / steps
        return gen_ep_loss, disc_ep_loss, ep_pesq
    
    def run_validation(self):
        gen_val_loss = 0
        disc_val_loss = 0
        val_pesq = 0
        steps = len(self.test_ds)
        self.model.eval()
        self.discriminator.eval()
        for _, batch in enumerate(self.test_ds):
            batch = self.preprocess_batch(batch)
            step_gen_loss, step_disc_loss, step_pesq = self.run_validation_step(batch)
            
            gen_val_loss += step_gen_loss
            disc_val_loss += step_disc_loss
            val_pesq += step_pesq

        gen_val_loss = gen_val_loss / steps
        disc_val_loss = disc_val_loss / steps
        val_pesq = val_pesq / steps

        
        return gen_val_loss, disc_val_loss, val_pesq

    def run_validation_step(self, batch):
        agent = SpeechEnhancementAgent(window=self.win_len // 2, 
                                       n_fft=self.n_fft,
                                       hop=self.hop,
                                       gpu_id=self.gpu_id)
        agent.set_batch(batch)
        generator_outputs = self.forward_generator_loop(agent)
        
        #calculate generator_loss
        gen_loss = self.calculate_generator_loss(generator_outputs)

        #calculate discriminator loss
        disc_loss, pesq = self.calculate_discriminator_loss(generator_outputs)
        
        pesq = original_pesq(pesq)
        return gen_loss.item(), disc_loss.item(), pesq  
    
    def train(self, args):
        best_val_gen_loss = 9999
        best_val_disc_loss = 9999
        for epoch in range(args.epochs):
            #Run training loop
            gen_ep_loss, disc_ep_loss, ep_pesq = self.train_one_epoch()

            wandb.log({
                "Epoch":epoch,
                "ep_gen_loss":gen_ep_loss,
                "ep_disc_loss":disc_ep_loss,
                "ep_train_pesq":ep_pesq
            })
            print(f"Epoch:{epoch}, Train_G_Loss:{gen_ep_loss}, Train_D_Loss:{disc_ep_loss}, Train_PESQ:{ep_pesq}")
            
            #Run validation loop
            gen_val_loss, disc_val_loss, val_pesq = self.run_validation()

            wandb.log({
                "Epoch":epoch,
                "val_gen_loss":gen_val_loss,
                "val_disc_loss":disc_val_loss,
                "val_train_pesq":val_pesq
            })
            print(f"Epoch:{epoch}, Val_G_Loss:{gen_val_loss}, Val_D_Loss:{disc_val_loss}, Val_PESQ:{val_pesq}")

            if gen_val_loss <= best_val_gen_loss or disc_val_loss <= best_val_disc_loss:
                best_val_gen_loss = min(best_val_gen_loss, gen_val_loss)
                best_val_disc_loss = min(best_val_disc_loss, gen_val_loss)

                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)

                self.save_model(path_root=args.save_model_dir,
                                exp=args.exp,
                                epoch=epoch,
                                pesq=val_pesq)
                
            self.scheduler_G.step()
            self.scheduler_D.step()


    
    def save_model(self, path_root, exp, epoch, pesq):
        """
        Save model at path_root
        """
        checkpoint_prefix = f"{exp}_PESQ_{pesq}_epoch_{epoch}.pt"
        path = os.path.join(path_root, checkpoint_prefix)
        
        save_dict = {'generator_state_dict':self.model.state_dict(), 
                     'discriminator_state_dict':self.discriminator.state_dict(),
                     'optimizer_G_state_dict':self.optimizer.state_dict(),
                     'optimizer_D_state_dict':self.optimizer_disc.state_dict(),
                     'scheduler_G_state_dict':self.scheduler_G.state_dict(),
                     'scheduler_D_state_dict':self.scheduler_D.state_dict(),
                    }
        
        torch.save(save_dict, path)
        print(f"checkpoint:{checkpoint_prefix} saved at {path}")

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, args):
    if args.parallel:
        ddp_setup(rank, world_size)
        if rank == 0:
            print(args)
            available_gpus = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            print(f"Available gpus:{available_gpus}")
    
        train_ds, test_ds = dataloader.load_data(
            args.data_dir, args.batch_size, 1, args.cut_len, parallel=True
        )
    else:
        train_ds, test_ds = dataloader.load_data(
            args.data_dir, args.batch_size, 1, args.cut_len, parallel=False
        )
    
    trainer = FrameLevelTrainer(train_ds=train_ds, 
                                test_ds=test_ds, 
                                win_len=args.win_len, 
                                samples=args.samples, 
                                batchsize=args.batchsize, 
                                parallel=args.parallel, 
                                gpu_id=rank, 
                                pretrain=args.pretrain,
                                resume_pt=args.pt)
    trainer.train(args)
    if args.parallel:
        destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    if args.parallel:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    elif args.gpu:
        main(0, world_size, args)
    else:
        main(None, world_size, args)