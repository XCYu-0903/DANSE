<div align="center">
  <h1>DANSE</h1>

  <img src="assets/LOGO.png" width="400" alt="DANSE logo">

  <p>
    <strong>Xincheng Yu, Jingxue Huang, Jian Zhang, Dongyue Guo, Jianwei Zhang, and Yi Lin*</strong>
  </p>

  <p>
    <a href="#"><img src="https://img.shields.io/badge/Paper-coming_soon-blue?style=for-the-badge" alt="Paper"></a>
    <a href="https://xcyu-0903.github.io/DANSE-demo/"><img src="https://img.shields.io/badge/Demo-online-green?style=for-the-badge" alt="Demo"></a>
    <a href="https://github.com/XCYu-0903/DANSE"><img src="https://img.shields.io/badge/Code-DANSE-B19CD9?style=for-the-badge" alt="Code"></a>
    <a href="https://huggingface.co/datasets/Ediethia/Libri-AudioEvent"><img src="https://img.shields.io/badge/Dataset-Libri--AudioEvent-orange?style=for-the-badge" alt="Dataset"></a>
  </p>

  <p><strong>DANSE</strong> is pronounced as <em>‘dance’</em> (/dɑːns/).</p>
</div>

## Architecture

<div align="center">
  <img src="assets/DANSE.png" width="900" alt="DANSE architecture">
</div>


## Structure

- <img src="assets/not_released.svg" width="16" height="16" align="absmiddle" alt="not released"> `train.py`: Training entry. It supports training from scratch and resuming from split checkpoints.
- <img src="assets/released.svg" width="16" height="16" align="absmiddle" alt="not released"> `config.json`: Default config template.
- <img src="assets/not_released.svg" width="16" height="16" align="absmiddle" alt="not released"> `model/`: DANSE backbone, which will be made available concurrently with the release of `train.py`.
- <img src="assets/not_released.svg" width="16" height="16" align="absmiddle" alt="not released"> `g_best`: Released checkpoint, which will be made available concurrently with the release of `train.py`.
- <img src="assets/not_released.svg" width="16" height="16" align="absmiddle" alt="not released"> `inference.py`: Achieving both speech enhancement and noise estimation, which will be made available concurrently with the release of `train.py`. 


## Acknowledgements

We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet) and [CLAP](https://github.com/LAION-AI/CLAP) to implement this.

## Contact Us

If you are interested in leaving a message to our research team, feel free to email xinchengyu@alu.scu.edu.cn.

<p align="center">
  <img src="assets/scu_logo.png" height="130" align="middle" alt="SCU logo">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/wiseatc_logo.png" height="120" align="middle" alt="WiseATC logo">
</p>
