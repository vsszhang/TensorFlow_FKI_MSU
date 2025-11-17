# NNS Car Detection Task

🚀 This is an updated task project, uisng the CIFAR-10 to traning the model.
⚠️ This project using `uv` python scaffold.

## Step 0: Traning the model with CIFAR-10
🤖 Using the CIFAR-10 to train the model. After runing the following command, you will get an model file `cifar10_cnn.keras` in the directory `models/`.

```bash
uv run train_cifar10.py
```

## Step 1: Read the original image and check the size
📈 This script will help you load the image file `data/car.jpg`, print image size and show the image.

```bash
uv run main.py
```

## Step 2: Split image into tiles
✂️ Following splited rules to split the image, the tiles will transfer to the output directory `output/split/`.

```bash
uv run split_image.py
```

## Step 3: Reasoning patch
🧠 Using the model to reasoning, judging what is `car` or `truck`. The reasoning reason is output into the diretory `output/se/ected`.

`--thr` command flag is value of threshold.
```bash
uv run infer_tiles.py --thr 0.9
```

## Step 4: Visualize the output
👀 Visualize the output, the target image file will output as the `output/merged_result.png`.

```bash
uv run merge_and_visualize.py
```