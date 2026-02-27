# Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining
![Framework](assets/framework.png)

## 📌 Overview
We presented Keyframe-Chaining VLA, resolving non-Markovian ambiguity via Sparse Semantic History. Our Task-Modulated KSM extracts event-driven keyframes to efficiently ground long-horizon dependencies, achieving a 92.0% success rate on our ManiSkill Benchmark and robust real-world performance.

## 🛠️ Preparation
Here we provide a conda environment setup for the project.
```bash
   # clone the repository
    git clone https://github.com/cyp123cyp/KC-VLA.git
    cd KC-VLA
    
   conda create -n kcvla python=3.10
   conda activate kcvla
   # install dependencies
   pip install -r requirements.txt
   # Install ffmpeg (required only for torchcodec(real-bot))
   conda install -c conda-forge ffmpeg==7.1.1
   # Install Flash Attention (required for efficient VLA inference)
   pip install --no-build-isolation flash-attn==2.7.1.post4
```

## Dataset
All simulation-based datasets and experiments in this project are conducted using the ManiSkill environment.

The [ManiSkill](https://github.com/haosulab/ManiSkill) simulation environment should be set up following the [official installation guide](https://github.com/haosulab/ManiSkill#installation).
Please installing ManiSkill in a separate conda environment following the official guide.

While we utilize [ManiSkill](https://github.com/haosulab/ManiSkill) as our simulation backbone, all long-horizon, memory-dependent datasets used in this project are custom-generated to support our research on Keyframe-Chaining VLA.

1. Generate via Our Collection Pipeline
You can reproduce our data collection process or generate new trajectories for our 4 custom tasks by running our pipeline:
```bash
    bash mani_skill/examples/motionplanning/panda/collectdata.sh
```
2. Download from Hugging Face
You can also download from [Hugging Face](https://huggingface.co/datasets/furry123/ManiSkill-Memory-dependence).

## Training
1. Training the keyframe selection module
```bash
   python keyframe_selection_module/train_stage1.py
   python keyframe_selection_module/train_stage2.py
```
2. Training Keyframe Chaining VLA
```bash
   python scripts/gr00t_finetune.py
```

## Evaluation
The evaluation is deployed in a client–server architecture, where the policy model runs as a service and the ManiSkill environment interacts with it as a client. To evaluate the model on ManiSkill, follow the steps below.

Step 1: Launch the Policy Service(use kcvla Environment)
```bash
    # Start the policy inference service
    python scripts/inference_service.py
```

Step 2: Launch the ManiSkill Client(use Maniskill Environment)
```bash
    # Start the ManiSkill client
    python evaluate/eval_for_maniskill.py
```

## Acknowledgments
[ManiSkill](https://github.com/haosulab/ManiSkill) - Original framework
[SAPIEN](https://sapien.ucsd.edu/) - Physics simulation engine
[Issac GR00T](https://github.com/NVIDIA/Isaac-GR00T) - VLA base model

## 📝 Citation
If you find this project or the custom ManiSkill benchmark useful for your research, please consider citing:

```bibtex
@article{KC-VLA,
  title={Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining},
  author={Yipeng Chen and Wentao Tan and Lei Zhu and Fengling Li and Jingjing Li and Guoli Yang and Heng Tao Shen},
  journal={arXiv preprint arXiv},
  year={2026},
}