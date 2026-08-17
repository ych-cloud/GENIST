# GENIST

## Predicting comprehensive spatial gene expression landscapes from H&E images using gene network context

![GENIST overview](figures/fig1.png)

GENIST is a conditional generative framework for predicting spatial gene expression from H&E images. It uses histology-derived representations together with gene-network context and supports both spot-level and single-cell experiments.

This repository contains the model code, data-preparation scripts and commands used to reproduce the experiments reported in the paper.

The codebase for this study is publicly available at [https://github.com/ych-cloud/GENIST](https://github.com/ych-cloud/GENIST).

## System requirements

The code was developed for a CUDA-enabled Python environment. The supplied environment uses PyTorch 2.3.1, torchvision 0.18.1 and CUDA 11.8. The package list does not record the Python interpreter version; Python 3.10 is used in the example below.

## Installation

Run all commands from the repository root.

~~~bash
conda create -n genist python=3.10
conda activate genist
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
~~~

The supplied paper-code environment uses PyTorch 2.3.1 and torchvision 0.18.1 with CUDA 11.8. The extra index URL is required because these CUDA wheels are not hosted on the default PyPI index.

~~~cmd
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
~~~

This requirements file targets the CUDA 11.8 environment. For CPU-only execution, replace the two CUDA-specific PyTorch lines with CPU-compatible torch and torchvision versions before installation. The repository does not include private data, foundation-model weights, HoverNet, UNI, or CONCH. Install or download these components separately according to their licences and access requirements.

The external encoders used by the preprocessing scripts are:

- UNI for pathology-image embeddings.
- CONCH for pathology-image embeddings.
- HoverNet for H&E nuclei segmentation in the single-cell pipeline.
- HEST-1k for the public spot-level data.

The H&E/Xenium registration command additionally imports insitupy. This package was not present in the supplied environment export and must be installed separately if that step is used.

## Repository structure

~~~text
GENIST/                              Model, data loading, training, sampling, evaluation
Gene_Order/                          Gene-ordering code and bundled RegNetwork graphs
spot_datasets_processing/            HEST/HER2ST download and spot preprocessing
single_cell_datasets_processing/     Xenium/H&E registration and single-cell preprocessing
train.py                             Training entry point
sample.py                            Sampling entry point
evaluate.py                          Evaluation entry point
figures/                             Figures shown in the paper
datasets/                            Local data and derived features; ignored by Git
experiments/                         Local checkpoints and results; ignored by Git
requirements.txt                     Runtime dependencies
~~~

The data and experiment directories are intentionally kept out of version control. Create them locally using the layouts below.

## Usage

The workflow has two experimental settings: spot-level prediction on HER2ST and single-cell prediction on Xenium data.

### 1. Spot-level prediction: HER2ST

#### 1. Download HEST-1k data

Set a Hugging Face token if the selected HEST components require authentication.

~~~cmd
set HF_TOKEN=your_token
python spot_datasets_processing/download_hest1k.py --subset her2st --components wsis st --output-dir datasets/her2st --token-env HF_TOKEN
~~~

On PowerShell, use:

~~~powershell
$env:HF_TOKEN = "your_token"
python spot_datasets_processing/download_hest1k.py --subset her2st --components wsis st --output-dir datasets/her2st --token-env HF_TOKEN
~~~

Use the dry-run option before a large download:

~~~bash
python spot_datasets_processing/download_hest1k.py --subset her2st --components wsis st --output-dir datasets/her2st --dry-run
~~~

#### 2. Select and order genes

The following command selects the top 200 genes from the HER2ST training data. Change the value of target-genes when running another gene setting.

~~~bash
python spot_datasets_processing/build_her2st_gene_list.py ^
  --dataset-root datasets/her2st ^
  --output-root datasets/her2st/derived_features ^
  --target-genes 200 ^
  --gene-list-filename genes_selected_top200.txt ^
  --slides-filename samples_her2st.txt ^
  --ranking-filename genes_selection_stats_top200.csv
~~~

The multiline commands use Windows CMD caret continuation. In PowerShell, replace each caret with the PowerShell line-continuation character; on Linux or macOS, replace each caret with a backslash.

Order the selected genes with the bundled human RegNetwork graph:

~~~bash
python Gene_Order/gene_order_filtered.py ^
  --graph Gene_Order/RegNetwork/graph_human_core.graphml ^
  --selected-genes datasets/her2st/derived_features/genes_selected_top200.txt ^
  --output datasets/her2st/derived_features/genes_ordered_top200_regnetwork.txt
~~~

#### 3. Extract spot image embeddings

UNI and CONCH are selectable with the encoders option. The example below extracts both embeddings.

~~~bash
python spot_datasets_processing/extract_her2st_embeddings.py ^
  --dataset-root datasets/her2st ^
  --output-root datasets/her2st/derived_features ^
  --slides-file datasets/her2st/derived_features/samples_her2st.txt ^
  --encoders uni conch ^
  --device cuda:0 ^
  --num-augments 7
~~~

If CONCH weights are not in the environment, provide them explicitly:

~~~bash
python spot_datasets_processing/extract_her2st_embeddings.py ^
  --dataset-root datasets/her2st ^
  --output-root datasets/her2st/derived_features ^
  --encoders conch ^
  --conch-weights path/to/conch_weights.bin ^
  --device cuda:0
~~~

#### 4. Train, sample, and evaluate

The training entry point automatically creates sequential run directories under experiments/spot.

~~~bash
python train.py ^
  --mode spot ^
  --data-path datasets/her2st ^
  --feature-root datasets/her2st/derived_features ^
  --gene-order-file genes_ordered_top200_regnetwork.txt ^
  --sample-id-file samples_her2st.txt ^
  --spot-condition-sources uni conch ^
  --slide-out SPA123 ^
  --experiment-root experiments ^
  --device cuda:0
~~~

Replace SPA123 with the held-out slide used in the experiment. To resume the newest run, add resume latest.

~~~bash
python sample.py ^
  --mode spot ^
  --checkpoint experiments/spot/run_000/checkpoints/checkpoint_step_0000000.pt ^
  --data-path datasets/her2st ^
  --feature-root datasets/her2st/derived_features ^
  --gene-order-file genes_ordered_top200_regnetwork.txt ^
  --slide-out SPA123 ^
  --spot-condition-sources uni conch ^
  --output-dir experiments/spot/run_000/predictions ^
  --device cuda:0
~~~

Evaluate a generated expression tensor:

~~~bash
python evaluate.py ^
  --mode spot ^
  --prediction-file experiments/spot/run_000/predictions/expression_samples_checkpoint_step_0000000_n16.pt ^
  --data-path datasets/her2st ^
  --feature-root datasets/her2st/derived_features ^
  --gene-order-file genes_ordered_top200_regnetwork.txt ^
  --slide-out SPA123 ^
  --output-dir experiments/spot/run_000/evaluations/spot_spa123
~~~

Use python train.py --help, python sample.py --help, and python evaluate.py --help to inspect all model and evaluation options.

### 2. Single-cell prediction

The single-cell pipeline follows this order:

1. Register the H&E image to Xenium coordinates.
2. Segment H&E nuclei with HoverNet.
3. Rasterize Xenium nucleus boundaries.
4. Build the Xenium cell-by-gene matrix.
5. Match H&E nuclei to Xenium nuclei.
6. Extract cell-centred H&E patches and masks.
7. Extract UNI embeddings and create spatial folds.
8. Optionally add cell-type and neighbourhood conditions.
9. Train, sample, and evaluate GENIST.

Use one sample directory per Xenium sample:

~~~text
datasets/single_cell/<sample_id>/
├── raw/
│   ├── xenium_run/
│   └── he_image.tif
├── processed/
└── derived_features/
~~~

The following commands are a template. Replace all paths in angle brackets with paths for the selected sample.

#### 1. Register H&E to Xenium

~~~bash
python single_cell_datasets_processing/register_he_insitupy.py ^
  --xenium-dir datasets/single_cell/<sample_id>/raw/xenium_run ^
  --he-image datasets/single_cell/<sample_id>/raw/he_image.tif
~~~

Save or export the registered H&E image as:

~~~text
datasets/single_cell/<sample_id>/processed/he_image_registered.tif
~~~

#### 2. Segment H&E nuclei with HoverNet

Run the three steps in order. Step 2 requires a local HoverNet checkout and model weights.

~~~bash
python single_cell_datasets_processing/segment_he_nuclei.py ^
  --fp_he_img datasets/single_cell/<sample_id>/processed/he_image_registered.tif ^
  --dir_output datasets/single_cell/<sample_id>/processed ^
  --step 1

python single_cell_datasets_processing/segment_he_nuclei.py ^
  --fp_he_img datasets/single_cell/<sample_id>/processed/he_image_registered.tif ^
  --dir_hovernet <path_to_hovernet> ^
  --dir_output datasets/single_cell/<sample_id>/processed ^
  --step 2

python single_cell_datasets_processing/segment_he_nuclei.py ^
  --fp_he_img datasets/single_cell/<sample_id>/processed/he_image_registered.tif ^
  --dir_output datasets/single_cell/<sample_id>/processed ^
  --fp_out_seg he_image_nuclei_seg.tif ^
  --step 3
~~~

The merge step writes the full-resolution segmentation and a micron-resolution file named he_image_nuclei_seg_microns.tif.

#### 3. Build Xenium segmentation and expression matrix

~~~bash
python single_cell_datasets_processing/build_xenium_nuclei_segmentation.py ^
  --fp-boundaries datasets/single_cell/<sample_id>/raw/xenium_run/analysis/nucleus_boundaries.csv.gz ^
  --fp-he-img datasets/single_cell/<sample_id>/processed/he_image_registered.tif ^
  --dir-output datasets/single_cell/<sample_id>/processed ^
  --fp-out-nuclei-seg xenium_nuclei_segmentation.tif ^
  --fp-ids-out xenium_cell_ids_dict.csv

python single_cell_datasets_processing/build_xenium_cell_gene_matrix.py ^
  --dir-feature-matrix datasets/single_cell/<sample_id>/raw/xenium_run/cell_feature_matrix ^
  --dir-output datasets/single_cell/<sample_id>/processed ^
  --fp-genes genes_source.txt ^
  --fp-out-matrix expression_raw.csv
~~~

#### 4. Match cells and extract patches

Both segmentation masks must be at the same micron-scale resolution.

~~~bash
python single_cell_datasets_processing/match_corresponding_cells.py ^
  --fp-seg-hist he_image_nuclei_seg_microns.tif ^
  --fp-seg-xenium xenium_nuclei_segmentation.tif ^
  --fp-cgm expression_raw.csv ^
  --dir-output datasets/single_cell/<sample_id>/processed ^
  --fp-out-matched-nuclei matched_nuclei.csv

python single_cell_datasets_processing/extract_single_cell_patches.py ^
  --fp-hist datasets/single_cell/<sample_id>/processed/he_image_registered.tif ^
  --fp-seg-hist datasets/single_cell/<sample_id>/processed/he_image_nuclei_seg_microns.tif ^
  --fp-cells-csv datasets/single_cell/<sample_id>/processed/matched_nuclei_filtered.csv ^
  --output-dir datasets/single_cell/<sample_id>/processed/single_cell_patches
~~~

The filtered expression matrix is written to processed/expression_raw_filtered.csv. The patch directory contains patches, masks, and patch_metadata.csv.

#### 5. Extract UNI embeddings and build folds

~~~bash
python single_cell_datasets_processing/build_uni_embeddings.py ^
  --patch-metadata datasets/single_cell/<sample_id>/processed/single_cell_patches/patch_metadata.csv ^
  --patch-root datasets/single_cell/<sample_id>/processed/single_cell_patches ^
  --expr-csv datasets/single_cell/<sample_id>/processed/expression_raw_filtered.csv ^
  --fp-seg-hist datasets/single_cell/<sample_id>/processed/he_image_nuclei_seg_microns.tif ^
  --output-dir datasets/single_cell/<sample_id>/derived_features ^
  --slide-name <sample_id> ^
  --gene-list-filename genes_selected.txt ^
  --sample-list-filename samples_single_cell.txt ^
  --device cuda:0
~~~

This step creates fold_1, fold_2, and later fold directories under derived_features. Each fold contains train and val data, expression.csv, coordinates, and UNI embeddings.

If extra cell-type and neighbourhood conditions are needed, run the following command once per fold:

~~~bash
python single_cell_datasets_processing/build_extra_condition.py ^
  --fold-dir datasets/single_cell/<sample_id>/derived_features/fold_1 ^
  --cell-type-csv datasets/single_cell/<sample_id>/raw/cell_types.csv ^
  --k-neighbors 8
~~~

#### 6. Train, sample, and evaluate single-cell data

~~~bash
python train.py ^
  --mode single_cell ^
  --data-path datasets/single_cell/<sample_id> ^
  --feature-root datasets/single_cell/<sample_id>/derived_features ^
  --gene-order-file genes_selected.txt ^
  --fold 1 ^
  --slide-name <sample_id> ^
  --experiment-root experiments ^
  --device cuda:0
~~~

The sample and evaluation commands use the same data paths and fold:

~~~bash
python sample.py ^
  --mode single_cell ^
  --checkpoint experiments/single_cell/run_000/checkpoints/checkpoint_step_0000000.pt ^
  --data-path datasets/single_cell/<sample_id> ^
  --feature-root datasets/single_cell/<sample_id>/derived_features ^
  --gene-order-file genes_selected.txt ^
  --fold 1 ^
  --split val ^
  --slide-name <sample_id> ^
  --output-dir experiments/single_cell/run_000/predictions ^
  --device cuda:0

python evaluate.py ^
  --mode single_cell ^
  --prediction-file experiments/single_cell/run_000/predictions/expression_samples_checkpoint_step_0000000_n16.pt ^
  --data-path datasets/single_cell/<sample_id> ^
  --feature-root datasets/single_cell/<sample_id>/derived_features ^
  --gene-order-file genes_selected.txt ^
  --fold 1 ^
  --split val ^
  --slide-name <sample_id> ^
  --output-dir experiments/single_cell/run_000/evaluations/fold_1
~~~

Add use-extra-condition to train.py, sample.py, and evaluate.py only when extra_cond.npy has been generated for the corresponding fold.

## Output naming

The repository uses the following names so that preprocessing, training, and evaluation can be connected without manual renaming:

~~~text
genes_selected_top<N>.txt
genes_selection_stats_top<N>.csv
genes_ordered_top<N>_regnetwork.txt
samples_her2st.txt
samples_single_cell.txt
run_000/
checkpoints/checkpoint_step_<step>.pt
predictions/expression_samples_checkpoint_step_<step>_n<repetitions>.pt
evaluations/<evaluation_name>/
~~~

Training runs are created as experiments/spot/run_### or experiments/single_cell/run_###. Each run stores run_config.json, dataset_manifest.json, training.log, checkpoints, predictions, and evaluation outputs.

## Data & model availability

Raw datasets, private pathology images, H&E/Xenium files, pretrained weights, and generated tensors are not redistributed in this repository. Obtain public data from the original sources and place local files under datasets/. Do not commit patient-level or otherwise restricted data.

## Citation

If you use GENIST, please cite the accompanying paper:

~~~text
GENIST: predicting comprehensive spatial gene expression landscapes from H&E images using gene network context.
~~~

Full citation information will be added after publication.

## License

Add the project licence and author information before making the repository public.
