# Energy-based Models are Zero-Shot Planners for Compositional Scene Rearrangement

By [Nikolaos Gkanatsios*](https://nickgkan.github.io/), [Ayush Jain*](https://ayushjain1144.github.io/), [Zhou Xian](https://www.zhou-xian.com/), [Yunchu Zhang](https://github.com/YunchuZhang), [Christopher G. Atkeson](http://www.cs.cmu.edu/~cga/),  [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/).

Official implementation of ["Energy-based Models are Zero-Shot Planners for Compositional Scene Rearrangement"](https://arxiv.org/abs/2304.14391), accepted by RSS 2023.

![teaser](https://ebmplanner.github.io/static/images/model.png)


## Install

### Requirements
We showcase the installation for CUDA 11.1 and torch==1.10.2, which is what we used for our experiments.

- conda create -n "srem" python=3.8
- conda activate srem
- pip install -r requirements.txt
- pip install -U torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cu111
- sh scripts/init.sh (Make sure you have gcc>=5.5.0)


### Data Preparation

For generating simulation data for our benchmarks, execute
- python demos.py

This will generate data for cliport tasks, spatial relations, shapes and compositional benchmarks. If you want to generate data for a specific benchmark, you can comment out the rest from `task_list`

Download all the needed checkpoints:
- wget https://zenodo.org/record/8114634/files/checkpoints.zip?download=1


## Usage

We provide scripts for training goal conditioned transporter and evaluating our full model on various benchmarks in scripts folder. 
- sh scripts/run_train_composition_one_step.sh # composition-one-step benchmark
- sh scripts/run_train_composition_group.sh # composition-group benchmark
- sh scripts/run_train_single_relations.sh # relations benchmark
- sh scripts/run_train_shapes.sh # shapes benchmark
- sh scripts/run_train_cliport.sh # cliport benchmark

If you followed the Data Preparation section, you should have access to all the checkpoints needed to do the evaluations and reproduce the results.


## Training Individual Modules

### Parser 
For training the language parser, first generate the language data by running:
- python data/create_cliport_programs.py

This will create a json file with paired language sentence and expected program. 

For actually training it, you can run:
- sh scripts/train_parser.sh

### EBM
To train the EBMs:
- sh scripts/train_ebm_script.sh

This will train the EBMs on all different concepts we use in our paper. You can isolate a command from this script to train on your concept of interest. Use the --eval flag to run inference and visualize generated samples.

### Grounding Model
We train the grounding model using the publically available code of [BUTD-DETR](https://github.com/nickgkan/butd_detr)

## Vizualizations and Results
Please check our [website](https://ebmplanner.github.io/) for qualitative results and a quick overview of this project. 

## Acknowledgements
Parts of this code were based on the codebase of [CLIPort](https://github.com/cliport/cliport). The code for grounding module is borrowed from [BUTD-DETR](https://github.com/nickgkan/butd_detr). Parts of the EBM training code were based on the codebase of [compose-visual-relations](https://github.com/nanlliu/compose-visual-relations).


## Citing SREM
If you find SREM useful in your research, please consider citing:
```bibtex
@article{gkanatsios2023energy,
  title={Energy-based models as zero-shot planners for compositional scene rearrangement},
  author={Gkanatsios, Nikolaos and Jain, Ayush and Xian, Zhou and Zhang, Yunchu and Atkeson, Christopher and Fragkiadaki, Katerina},
  journal={arXiv preprint arXiv:2304.14391},
  year={2023}
}
    
```

