# Monocular depth assisted tracking by detection

## Environment

Create the Conda environment
```
conda env create -f env_py37_mac.yml
```

Activate the Conda environment
```
conda activate py37-mdtd
```

## Prepare the KITTI image folders

Go to `data/kitti/20110926/image/` and create the links to the `image_02/data`
folders from the KITTI sequences taken with the `2011_09_26` date tag.

## Run the inference pipeline

Assume the Conda environment is correctly activated
```
cd inference_pipeline
python main.py
```
Modify `config.json` for input/output settings.
The output folder will contains an aggregated `main.avi` for visualization.

## Acknowledgment 

Great appreciation from the pretrained models and the training details from:
- Object detection (v1.13.0): [GitHub link](https://github.com/tensorflow/models/tree/v1.13.0/research/object_detection)
- Struct2Depth (v1.13.0): [GitHub link](https://github.com/tensorflow/models/tree/v1.13.0/research/struct2depth)

KITTI dataset source: [Dataset link](http://www.cvlibs.net/datasets/kitti/raw_data.php)
