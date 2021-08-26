# Scene Graph Benchmark in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

Paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949) has been accepted by CVPR 2020 (Oral).

## Recent Updates
- [ ] TODO: Deployment on sagemaker
## Contents

1. [Quick Start](#Quick Start)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Metrics and Results for our Toolkit](METRICS.md)
    - [Explanation of R@K, mR@K, zR@K, ng-R@K, ng-mR@K, ng-zR@K, A@K, S2G](METRICS.md#explanation-of-our-metrics)
    - [Output Format](METRICS.md#output-format-of-our-code)
    - [Reported Results](METRICS.md#reported-results)
5. [Faster R-CNN Pre-training](#pretrained-models)
6. [Training on Scene Graph Generation](#perform-training-on-scene-graph-generation)
7. [Evaluation on Scene Graph Generation](#Evaluation)
8. [Detect Scene Graphs on Your Custom Images](#SGDet-on-custom-images)
9. [Visualize Detected Scene Graphs of Custom Images](#Visualize-Detected-SGs-of-Custom-Images)



## Quick Start

Check [customs.ipynb](customs.ipynb) to quick start. (No need to view the process behind)

## Installation
`sh gcc.sh` to install gcc==7.3.0.
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Metrics and Results **(IMPORTANT)**
Explanation of metrics in our toolkit and reported results are given in [METRICS.md](METRICS.md)

## Pretrained Models

Please download the [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ),  extract all the files to the directory `/home/username/checkpoints/pretrained_faster_rcnn`. To train your own Faster R-CNN model, please follow the next section.

The above pretrained Faster R-CNN model achives 38.52/26.35/28.14 mAp on VG train/val/test set respectively.

## Faster R-CNN pre-training
The following command can be used to train your own Faster R-CNN model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/kaihua/checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```
where ```CUDA_VISIBLE_DEVICES``` and ```--nproc_per_node``` represent the id of GPUs and number of GPUs you use, ```--config-file``` means the config we use, where you can change other parameters. ```SOLVER.IMS_PER_BATCH``` and ```TEST.IMS_PER_BATCH``` are the training and testing batch size respectively, ```DTYPE "float16"``` enables Automatic Mixed Precision supported by [APEX](https://github.com/NVIDIA/apex), ```SOLVER.MAX_ITER``` is the maximum iteration, ```SOLVER.STEPS``` is the steps where we decay the learning rate, ```SOLVER.VAL_PERIOD``` and ```SOLVER.CHECKPOINT_PERIOD``` are the periods of conducting val and saving checkpoint, ```MODEL.RELATION_ON``` means turning on the relationship head or not (since this is the pretraining phase for Faster R-CNN only, we turn off the relationship head),  ```OUTPUT_DIR``` is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), ```SOLVER.PRE_VAL``` means whether we conduct validation before training or not.

## Perform training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols. 

For **Predicate Classification (PredCls)**, we need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For our predefined Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```
For [Unbiased-Causal-TDE](https://arxiv.org/abs/2002.11949) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```

### Customize Your Own Model
If you want to customize your own model, you can refer ```maskrcnn-benchmark/modeling/roi_heads/relation_head/model_XXXXX.py``` and ```maskrcnn-benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py```. You also need to add corresponding nn.Module in ```maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py```. Sometimes you may also need to change the inputs & outputs of the module through ```maskrcnn-benchmark/modeling/roi_heads/relation_head/relation_head.py```.

### The proposed Causal TDE on [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949)
As to the Unbiased-Causal-TDE, there are some additional parameters you need to know. ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE``` is used to select the causal effect analysis type during inference(test), where "none" is original likelihood, "TDE" is total direct effect, "NIE" is natural indirect effect, "TE" is total effect. ```MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE``` has two choice "sum" or "gate". Since Unbiased Causal TDE Analysis is model-agnostic, we support [Neural-MOTIFS](https://arxiv.org/abs/1711.06640), [VCTree](https://arxiv.org/abs/1812.01880) and [VTransE](https://arxiv.org/abs/1702.08319). ```MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER``` is used to select these models for Unbiased Causal Analysis, which has three choices: motifs, vctree, vtranse.

Note that during training, we always set ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE``` to be 'none', because causal effect analysis is only applicable to the inference/test phase.

### Examples of the Training Command
Training Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```
where ```GLOVE_DIR``` is the directory used to save glove initializations, ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log. Since we use the ```WarmupReduceLROnPlateau``` as the learning scheduler for SGG, ```SOLVER.STEPS``` is not required anymore.

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```


## Evaluation

### Examples of the Test Command
Test Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10028 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgcls-exmp OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```

### Examples of Pretrained Causal MOTIFS-SUM models
Examples of Pretrained Causal MOTIFS-SUM models on SGDet/SGCls/PredCls (batch size 12): [(SGDet Download)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs), [(SGCls Download)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781938&authkey=AO_ddcgNpVVGE-g), [(PredCls Download)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781937&authkey=AOzowl5-07RzJz4)

Corresponding Results (The original models used in the paper are lost. These are the fresh ones, so there are some fluctuations on the results. More results can be found in [Reported Results](METRICS.md#reported-results)):

Models |  R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
MOTIFS-SGDet-none   | 25.42 | 32.45 | 37.26 | 4.36 | 5.83 | 7.08 | 0.02 | 0.08 | 0.24
MOTIFS-SGDet-TDE    | 11.92 | 16.56 | 20.15 | 6.58 | 8.94 | 10.99 | 1.54 | 2.33 | 3.03
MOTIFS-SGCls-none   | 36.02 | 39.25 | 40.07 | 6.50 | 8.02 | 8.51 | 1.06 | 2.18 | 3.07
MOTIFS-SGCls-TDE    | 20.47 | 26.31 | 28.79 | 9.80 | 13.21 | 15.06 | 1.91 | 2.95 | 4.10
MOTIFS-PredCls-none | 59.64 | 66.11 | 67.96 | 11.46 | 14.60 | 15.84 | 5.79 | 11.02 | 14.74
MOTIFS-PredCls-TDE  | 33.38 | 45.88 | 51.25 | 17.85 | 24.75 | 28.70 | 8.28 | 14.31 | 18.04

## SGDet on Custom Images
Note that evaluation on custum images is only applicable for SGDet model, because PredCls and SGCls model requires additional ground-truth bounding boxes information. To detect scene graphs into a json file on your own images, you need to turn on the switch TEST.CUSTUM_EVAL and give a folder path that contains the custom images to TEST.CUSTUM_PATH. Only JPG files are allowed. The output will be saved as custom_prediction.json in the given DETECTED_SGG_DIR.

Test Example 1 : (SGDet, **Causal TDE**, MOTIFS Model, SUM Fusion) [(checkpoint)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

Test Example 2 : (SGDet, **Original**, MOTIFS Model, SUM Fusion) [(same checkpoint)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

The output is a json file. For each image, the scene graph information is saved as a dictionary containing bbox(sorted), bbox_labels(sorted), bbox_scores(sorted), rel_pairs(sorted), rel_labels(sorted), rel_scores(sorted), rel_all_scores(sorted), where the last rel_all_scores give all 51 predicates probability for each pair of objects. The dataset information is saved as custom_data_info.json in the same DETECTED_SGG_DIR.

## Visualize Detected SGs of Custom Images
To visualize the detected scene graphs of custom images, you can follow the jupyter note: [visualization/3.visualize_custom_SGDet.jpynb](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/visualization/3.visualize_custom_SGDet.ipynb). The inputs of our visualization code are custom_prediction.json and custom_data_info.json in DETECTED_SGG_DIR. They will be automatically generated if you run the above custom SGDet instruction successfully. Note that there may be too much trivial bounding boxes and relationships, so you can select top-k bbox and predicates for better scene graphs by change parameters box_topk and rel_topk. 

