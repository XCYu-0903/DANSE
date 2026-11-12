import os
import random
import torch
import torch.utils.data
import librosa
import torchaudio
from glob import glob

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True) # [B, F,T] (complex 64)
    stft_spec = torch.view_as_real(stft_spec)    # [B, F, T, 2] divided into real and imaginary
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))  # absolute value  sqrt(r^2+i^2)
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5)) # attn2
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav


def get_dataset_filelist(training_file, validation_file):
    def read_manifest(path):
        with open(path, 'r', encoding='utf-8') as file:
            return [line.split('|', 1)[0] for line in file.read().splitlines() if line]

    return read_manifest(training_file), read_manifest(validation_file)


def utterance_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return "_".join(stem.split("_")[-2:])


def pad_or_truncate(x, audio_length, is_repeated=False):
    # x : torch.tensor  [1, waveform]
    """Pad all audio to specific length."""
    cur_len = x.size(1)

    if cur_len >= audio_length: # [1,2,3,4,5,6,7,8,9,10,11,12] -> [1,2,3,4,5,6,7,8,9,10]
        return x[:, 0: audio_length]

    if is_repeated:  # [1,2,3] -> [1,2,3,1,2,3,1,2,3,1]
        repeat_times = audio_length // cur_len
        remain = audio_length % cur_len

        x_repeated = x.repeat(1, repeat_times)

        if remain > 0:
            x_repeated = torch.cat([x_repeated, x[:, :remain]], dim=1)

        return x_repeated

    else:  # [1,2,3] -> [1,2,3,0,0,0,0,0,0,0]
        pad_len = audio_length - cur_len
        return torch.nn.functional.pad(x, (0, pad_len), mode='constant')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, segment_size,
                sampling_rate, ac_sampling_rate=32000, ac_segments=10 * 32000,
                 is_multimodal=False, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.ac_sampling_rate = ac_sampling_rate  # 32000
        self.ac_segments = ac_segments
        self.is_multimodal = is_multimodal
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.ac_resampler = {}


        self.clean_map = {}
        self.noisy_map = {}
        clean_files = glob(os.path.join(self.clean_wavs_dir,'*.wav'))
        noisy_files = glob(os.path.join(self.noisy_wavs_dir,'*.wav'))

        for fname in clean_files:
            self.clean_map[utterance_id(fname)] = fname

        for fname in noisy_files:
            self.noisy_map[utterance_id(fname)] = fname

    def __getitem__(self, index):

        idname = self.audio_indexes[index]  # e.g., fileid_161
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(self.clean_map[idname], sr=self.sampling_rate)
            noisy_audio, _ = librosa.load(self.noisy_map[idname], sr=self.sampling_rate)
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[: length], noisy_audio[: length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse

        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.is_multimodal:
            clean_audio_ac = clean_audio
            noisy_audio_ac = noisy_audio
            if self.sampling_rate != self.ac_sampling_rate:
                sr_str = str(self.sampling_rate)
                if sr_str not in self.ac_resampler:
                    self.ac_resampler[sr_str] = torchaudio.transforms.Resample(
                        orig_freq=self.sampling_rate, new_freq=self.ac_sampling_rate
                    )
                clean_audio_ac = self.ac_resampler[sr_str](clean_audio)
                noisy_audio_ac = self.ac_resampler[sr_str](noisy_audio)
            clean_audio_ac = pad_or_truncate(clean_audio_ac, self.ac_segments)
            noisy_audio_ac = pad_or_truncate(noisy_audio_ac, self.ac_segments)

        audio_start = 0
        audio_end = clean_audio.size(1)  # 16000 sampling_rate original len e.g., 48000 = 3s
        valid_end = audio_end  # valid audio end

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                if clean_audio.size(1) >= self.sampling_rate * 10:
                    max_audio_start = self.sampling_rate * 10 - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio_end = audio_start + self.segment_size
                valid_end = audio_end
                clean_audio = clean_audio[:, audio_start: audio_start+self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start+self.segment_size]
            else:
                audio_start = 0
                audio_end = self.segment_size
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')
        if self.is_multimodal:
            return (clean_audio.squeeze(), noisy_audio.squeeze(), clean_audio_ac.squeeze(), noisy_audio_ac.squeeze(),
                    torch.tensor(audio_start), torch.tensor(audio_end), torch.tensor(valid_end))
        return clean_audio.squeeze(), noisy_audio.squeeze()

    def __len__(self):
        return len(self.audio_indexes)



if __name__ == '__main__':
    random.seed(1234)
    lens = [50000, 240000, 350000, 670000, 870000, 67000]
    for len in lens:
        max_audio_start = len - 32000
        audio_start = random.randint(0, max_audio_start)
        print(audio_start)
