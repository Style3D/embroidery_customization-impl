<h2 align="center">
  One-shot Embroidery Customization <br> via Contrastive LoRA Modulation
</h2>

<p align="center">
  <a href="">
    <img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'>
  </a>
  <a href="https://style3d.github.io/embroidery_customization">
    <img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'>
  </a>
  <a href="https://huggingface.co/Style3D/EmoLoRA">
    <img src="https://img.shields.io/badge/HuggingFace-EmoLoRA-yellow?logo=huggingface&logoColor=white" alt="HuggingFace Model">
  </a>
</p>

<p align="center">
  <img src="assets/images/representative_image.jpg" width="100%">
</p>


> Diffusion models have significantly advanced image manipulation techniques, and their ability to generate photorealistic images is beginning to transform retail workflows, particularly in presale visualization. Beyond artistic style transfer, the capability to perform fine-grained visual feature transfer is becoming increasingly important.
> Embroidery is a textile art form characterized by intricate interplay of diverse stitch patterns and material properties, which poses unique challenges for existing style transfer methods. To explore the customization for such fine-grained features, we propose a novel contrastive learning framework that disentangles fine-grained style and content features with a single reference image, building on the classic concept of image analogy. We first construct an image pair to define the target style, and then adopt a similarity metric based on the decoupled representations of pretrained diffusion models for style-content separation. Subsequently, we propose a two-stage contrastive LoRA modulation technique to capture fine-grained style features. In the first stage, we iteratively update the whole LoRA and the selected style blocks to initially separate style from content. In the second stage, we design a contrastive learning strategy to further decouple style and content through self-knowledge distillation. Finally, we build an inference pipeline to handle image or text inputs with only the style blocks.
> To evaluate our method on fine-grained style transfer, we build a benchmark for embroidery customization. Our approach surpasses prior methods on this task and further demonstrates strong generalization to three additional domains: artistic style transfer, sketch colorization, and appearance transfer.

## Getting Started

### Installation

1. Clone the repository

```
git clone https://github.com/Style3D/embroidery_customization-impl.git
cd embroidery_customization-impl
```

2. Create and activate a virtual environment (venv)

```
python3 -m venv venv
source venv/bin/activate
```

> On Windows:

```
venv\Scripts\activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

### Pretrained Model

We release the pretrained EmoLoRA model on HuggingFace:

👉 **HuggingFace Model:**  
https://huggingface.co/Style3D/EmoLoRA

### Inference

We provide an example inference workflow in `inference.ipynb`.
Open the notebook and run the cells to reproduce the example results.

```
jupyter notebook inference.ipynb
```
