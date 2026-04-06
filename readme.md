# Multi-Subject Driven Image Generation

A project exploring personalized image generation from reference photos. Given one or more subject images, the system generates novel scenes featuring those subjects while preserving their visual identity. This work extends [SISO (Single Image Subject Optimization)](https://github.com/YuvalMevorach/SISO) to multi-subject settings and compares against DreamBooth dataset baselines.

**Course:** CMU 10-623 Generative AI

**Authors:** Gorkem Yar, Aditya Ramesh, Bhavesh Sood

---

## Results

<table>
  <tr>
    <th>Subject</th>
    <th>Generated</th>
  </tr>
  <tr>
    <td align="center">
      <img src="image.png" width="300" /><br />
      Wolf plushie (reference)
    </td>
    <td align="center">
      <img src="image-2.png" width="300" /><br />
      Wolf plushie at the Eiffel Tower
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="image-3.png" width="300" /><br />
      Grey sloth plushie (reference)
    </td>
    <td align="center">
      <img src="image-4.png" width="300" /><br />
      Grey sloth plushie drinking coffee
    </td>
  </tr>
</table>

### Quantitative Evaluation

| Method | DINO | IR | FID | KID | LPIPS |
|---|---|---|---|---|---|
| SISO (baseline paper) | 0.48 | 0.53 | 149.2 | 0.002 | 0.18 |
| SISO (ours) | 0.29 | 0.40 | 245.8 | 0.004 | 0.737 |
| Multi-Subject SISO (ours) | 0.65 | 0.76 | 217.1 | 0.05 | 0.71 |

- **DINO / IR**: Subject identity preservation (cosine similarity via DINOv2 and IR features). Higher is better.
- **FID / KID**: Distribution-level image quality. Lower is better.
- **LPIPS**: Perceptual diversity of generated outputs. Evaluated on the DreamBooth dataset (30 subjects).

---

## Approaches

### 1. SISO Reimplementation

A clean, from-scratch reimplementation of the SISO pipeline. The method is optimization-based: LoRA weights are updated per generation using a combined loss of DINOv2 and IR feature cosine similarity between the reference subject and the generated output. Early stopping is applied when the loss plateaus.

### 2. Multi-Subject Masked Algorithm

Extends SISO by introducing GroundingDINO and SAM (Segment Anything Model) to produce per-subject masks in the generated image. Each subject's loss is computed only within its corresponding mask region, enabling independent identity optimization for multiple subjects in a single scene. In practice, mask quality was unreliable for complex multi-object compositions, limiting this approach's effectiveness.

### 3. Multi-Subject Alternate Training

A round-robin training strategy where each subject is optimized independently in alternating steps, all sharing a single LoRA layer. This avoids the mask dependency entirely and produces more stable, consistent identity preservation across subjects. This was the most effective multi-subject approach.

---

## Project Structure

```
.
├── Genai_project.ipynb              # Main notebook (evaluation pipeline)
├── SISO_original/                   # Fork of official SISO with generation scripts
│   ├── siso_generation_sdxl.py      #   SDXL-Turbo generation
│   ├── siso_generation_flux.py      #   FLUX.1-schnell generation
│   ├── siso_generation_sana.py      #   Sana generation
│   └── environment.yml              #   Conda environment specification
├── SISO_reimplemented/              # Clean reimplementation with SAM mask support
│   ├── GenAI_SISO_Reimplemented.ipynb
│   ├── dino_utils.py
│   ├── ir_features_utils.py
│   ├── mask_utils.py
│   └── general_utils.py
├── dreambooth/                      # DreamBooth dataset (30 subjects)
├── reports/                         # Final project report
└── README.md
```

---

## Setup

Create the conda environment from the provided specification:

```bash
conda env create -f SISO_original/environment.yml
conda activate siso
```

---

## Usage

**Single-subject generation (SDXL-Turbo):**

```bash
python SISO_original/siso_generation_sdxl.py
```

**Single-subject generation (FLUX.1-schnell):**

```bash
python SISO_original/siso_generation_flux.py
```

**Single-subject generation (Sana):**

```bash
python SISO_original/siso_generation_sana.py
```

**Evaluation pipeline:**

Open and run `Genai_project.ipynb` to reproduce the quantitative evaluation (DINO, IR, FID, KID, LPIPS) across all methods.

**Reimplemented SISO with multi-subject support:**

Open and run `SISO_reimplemented/GenAI_SISO_Reimplemented.ipynb`.

---

## Tech Stack

- **Python** / **PyTorch** -- core framework
- **HuggingFace Diffusers** -- diffusion model pipelines (SDXL-Turbo, FLUX.1-schnell, Sana)
- **PEFT** -- parameter-efficient fine-tuning (LoRA)
- **DINOv2** -- self-supervised vision features for identity loss
- **OpenCLIP** -- CLIP-based feature extraction
- **SAM (Segment Anything)** -- subject mask generation
- **GroundingDINO** -- open-vocabulary object detection for mask prompting
- **Kornia** -- differentiable image processing
- **Accelerate** -- mixed-precision and multi-GPU training

---

## Report

The full project report with detailed methodology, ablations, and analysis is available at:

[`reports/GenAI___Final_Report (1).pdf`](<reports/GenAI___Final_Report (1).pdf>)

---

## Acknowledgments

This project builds on [SISO: Single Image Subject Optimization](https://github.com/YuvalMevorach/SISO) by Mevorach et al. and uses the [DreamBooth](https://dreambooth.github.io/) dataset for evaluation.
