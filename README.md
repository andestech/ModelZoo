
## Introduction
AndesAIRE(TM) Neural Network Software Development Kit (hereafter NN SDK) is a collection of software tools for facilitating the execution of pretrained neural network (NN) models on Andes development platforms.

# Model Zoo for AndesAIRE NN SDK
This repository provides a collection of widely used neural-network models and the resources required by the NNPilot(R) submodule in AndesAIRE NN SDK for quantizing FP32 models.

Each model entry includes the deployment workflow using AndesAIRE NN SDK, covering data preparation, calibration, evaluation, and the associated evaluation methods.

Each model directory also stores the generated quantized TFL model, the corresponding test inputs, and the resulting test metrics generated from either the TFL model or the fake-quantized model in PyTorch format, enabling rapid reproduction of the subsequent hardware-deployment workflow.

## Model Information

<table style="width:50%;text-align: center">
  <tr>
    <th>Network Architecture</th>
    <th>Task</th>
    <th>Network</th>
  </tr>

  <tr>
    <td rowspan="18">CNN</td>
    <td rowspan="8">Image Classification</td>
    <td><a href="MobileNetV1/README.md">MobileNetV1</a></td>
  </tr>
  <tr>
    <td><a href="MobileNetV2/README.md">MobileNetV2</a></td>
  </tr>
  <tr>
    <td><a href="ResNet-8/README.md">ResNet-8</a></td>
  </tr>
  <tr>
    <td><a href="ResNet-50/README.md">ResNet-50</a></td>
  </tr>
  <tr>
    <td><a href="ShuffleNet-v2/README.md">ShuffleNet-v2</a></td>
  </tr>
  <tr>
    <td><a href="SqueezeNet-v1_1/README.md">SqueezeNet-v1_1</a></td>
  </tr>
  <tr>
    <td><a href="EfficientNet-Lite0/README.md">EfficientNet-Lite0</a></td>
  </tr>
  <tr>
    <td><a href="Inception-v2/README.md">Inception-v2</a></td>
  </tr>

  <tr>
    <td rowspan="3">Object Detection</td>
    <td><a href="YOLOv1-tiny/README.md">YOLOv1-tiny</a></td>
  </tr>
  <tr>
    <td><a href="YOLOv2-tiny/README.md">YOLOv2-tiny</a></td>
  </tr>
  <tr>
    <td><a href="MobileNetV1-SSD/README.md">MobileNetV1-SSD</a></td>
  </tr>

  <tr>
    <td rowspan="2">Keyword Spotting</td>
    <td><a href="BC-ResNet-8/README.md">BC-ResNet-8</a></td>
  </tr>
  <tr>
    <td><a href="DS-CNN-L/README.md">DS-CNN-L</a></td>
  </tr>
  <tr>
    <td rowspan="1">Visual Wake Words</td>
    <td><a href="MCUNet-VWW2/README.md">MCUNet-VWW2</a></td>
  </tr>
  <tr>
    <td rowspan="1">Speech Recogntion</td>
    <td><a href="Tiny-Wav2letter/README.md">Tiny-Wav2letter</a></td>
  </tr>
  <tr>
    <td rowspan="2">Face Detection</td>
    <td><a href="BlazeFace/README.md">BlazeFace</a></td>
  </tr>
  <tr>
    <td><a href="RetinaFace/README.md">RetinaFace</a></td>
  </tr>
  <tr>
    <td rowspan="1">Face Recognitionn</td>
    <td><a href="GhostFaceNet/README.md">GhostFaceNet</a></td>
  </tr>
  <tr>
    <td rowspan="3">RNN</td>
    <td rowspan="3">Noise Suppresion</td>
    <td><a href="RNNoise/README.md">RNNoise</a></td>
  </tr>
  <tr>
    <td><a href="DeepFilterNet/README.md">DeepFilterNet</a></td>
  </tr>
  <tr>
    <td><a href="DTLN/README.md">DTLN</a></td>
  </tr>
</table>

## Installation and Usage

### Installing AndesAIRE NN SDK
Contact your Andes representative to obtain access to the AndesAIRE NN SDK toolkit.

### Installing Model Zoo
Clone this repository to the desired path following the instructions in the __AndesAIRE NNPilot User Manual (UM267).pdf__.

### Running Example Model
Prepare the dataset according to the instructions in each model directory's README. 
For running the quantization workflow of a selected model, see the __AndesAIRE NN SDK Starter Manual (UM266).pdf__.

## Team
AndesAIRE NN SDK Model Zoo is a project maintained by Andes Technology Corporation, Inc.

## License
See the [LICENSE file](LICENSE.txt) for details.

