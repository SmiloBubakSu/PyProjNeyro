# PyProjNeyro
Built with simplicity in mind, ImageAI supports a list of state-of-the-art Machine Learning algorithms for image prediction, custom image prediction, object detection, video detection, video object tracking and image predictions trainings. ImageAI currently supports image prediction and training using 4 different Machine Learning algorithms trained on the ImageNet-1000 dataset. ImageAI also supports object detection, video detection and object tracking using RetinaNet, YOLOv3 and TinyYOLOv3 trained on COCO dataset. Finally, ImageAI allows you to train custom models for performing detection and recognition of new objects.

Eventually, ImageAI will provide support for a wider and more specialized aspects of Computer Vision

New Release : ImageAI 3.0.2

What's new:

PyTorch backend
TinyYOLOv3 model training
TABLE OF CONTENTS
🔳 Installation
🔳 Features
🔳 Documentation
🔳 Sponsors
🔳 Projects Built on ImageAI
🔳 High Performance Implementation
🔳 AI Practice Recommendations
🔳 Contact Developers
🔳 Citation
🔳 References
Installation
To install ImageAI, run the python installation instruction below in the command line:

Download and Install Python 3.7, Python 3.8, Python 3.9 or Python 3.10

Install dependencies

CPU: Download requirements.txt file and install via the command

pip install -r requirements.txt
or simply copy and run the command below

pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3
GPU/CUDA: Download requirements_gpu.txt file and install via the command

pip install -r requirements_gpu.txt
or smiply copy and run the command below

pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cu102 torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cu102 pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3
If you plan to train custom AI models, download requirements_extra.txt file and install via the command

pip install -r requirements_extra.txt
or simply copy and run the command below

pip install pycocotools@git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
Then run the command below to install ImageAI

pip install imageai --upgrade
Features
Image Classification

ImageAI provides 4 different algorithms and model types to perform image prediction, trained on the ImageNet-1000 dataset. The 4 algorithms provided for image prediction include MobileNetV2, ResNet50, InceptionV3 and DenseNet121. Click the link below to see the full sample codes, explanations and best practices guide.
>>> Get Started
Object Detection

ImageAI provides very convenient and powerful methods to perform object detection on images and extract each object from the image. The object detection class provides support for RetinaNet, YOLOv3 and TinyYOLOv3, with options to adjust for state of the art performance or real time processing. Click the link below to see the full sample codes, explanations and best practices guide.
>>> Get Started
Video Object Detection & Analysis

ImageAI provides very convenient and powerful methods to perform object detection in videos. The video object detection class provided only supports the current state-of-the-art RetinaNet. Click the link to see the full videos, sample codes, explanations and best practices guide.
>>> Get Started
Custom Classification model training

ImageAI provides classes and methods for you to train a new model that can be used to perform prediction on your own custom objects. You can train your custom models using MobileNetV2, ResNet50, InceptionV3 and DenseNet in 5 lines of code. Click the link below to see the guide to preparing training images, sample training codes, explanations and best practices.
>>> Get Started
Custom Model Classification

ImageAI provides classes and methods for you to run image prediction your own custom objects using your own model trained with ImageAI Model Training class. You can use your custom models trained with MobileNetV2, ResNet50, InceptionV3 and DenseNet and the JSON file containing the mapping of the custom object names. Click the link below to see the guide to sample training codes, explanations, and best practices guide.
>>> Get Started
Custom Detection Model Training

ImageAI provides classes and methods for you to train new YOLOv3 or TinyYOLOv3 object detection models on your custom dataset. This means you can train a model to detect literally any object of interest by providing the images, the annotations and training with ImageAI. Click the link below to see the guide to sample training codes, explanations, and best practices guide.
>>> Get Started
Custom Object Detection

ImageAI now provides classes and methods for you detect and recognize your own custom objects in images using your own model trained with the DetectionModelTrainer class. You can use your custom trained YOLOv3 or TinyYOLOv3 model and the **.json** file generated during the training. Click the link below to see the guide to sample training codes, explanations, and best practices guide.
>>> Get Started
Custom Video Object Detection & Analysis

ImageAI now provides classes and methods for you detect and recognize your own custom objects in images using your own model trained with the DetectionModelTrainer class. You can use your custom trained YOLOv3 or TinyYOLOv3 model and the **.json** file generated during the training. Click the link below to see the guide to sample training codes, explanations, and best practices guide.
>>> Get Started
Documentation
We have provided full documentation for all ImageAI classes and functions. Visit the link below:

Documentation - English Version https://imageai.readthedocs.io
Sponsors
Real-Time and High Performance Implementation
ImageAI provides abstracted and convenient implementations of state-of-the-art Computer Vision technologies. All of ImageAI implementations and code can work on any computer system with moderate CPU capacity. However, the speed of processing for operations like image prediction, object detection and others on CPU is slow and not suitable for real-time applications. To perform real-time Computer Vision operations with high performance, you need to use GPU enabled technologies.

ImageAI uses the PyTorch backbone for it's Computer Vision operations. PyTorch supports both CPUs and GPUs ( Specifically NVIDIA GPUs. You can get one for your PC or get a PC that has one) for machine learning and artificial intelligence algorithms' implementations.

Projects Built on ImageAI
AI Practice Recommendations
For anyone interested in building AI systems and using them for business, economic, social and research purposes, it is critical that the person knows the likely positive, negative and unprecedented impacts the use of such technologies will have. They must also be aware of approaches and practices recommended by experienced industry experts to ensure every use of AI brings overall benefit to mankind. We therefore recommend to everyone that wishes to use ImageAI and other AI tools and resources to read Microsoft's January 2018 publication on AI titled "The Future Computed : Artificial Intelligence and its role in society". Kindly follow the link below to download the publication.

https://blogs.microsoft.com/blog/2018/01/17/future-computed-artificial-intelligence-role-society

Contact Developer
Moses Olafenwa
Email: guymodscientist@gmail.com
Twitter: @OlafenwaMoses
Medium: @guymodscientist
Facebook: moses.olafenwa
John Olafenwa
Email: johnolafenwa@gmail.com
Website: https://john.aicommons.science
Twitter: @johnolafenwa
Medium: @johnolafenwa
Facebook: olafenwajohn
Citation
You can cite ImageAI in your projects and research papers via the BibTeX entry below.

@misc {ImageAI,
    author = "Moses",
    title  = "ImageAI, an open source python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities",
    url    = "https://github.com/OlafenwaMoses/ImageAI",
    month  = "mar",
    year   = "2018--"
}
References
Somshubra Majumdar, DenseNet Implementation of the paper, Densely Connected Convolutional Networks in Keras https://github.com/titu1994/DenseNet
Broad Institute of MIT and Harvard, Keras package for deep residual networks https://github.com/broadinstitute/keras-resnet
Fizyr, Keras implementation of RetinaNet object detection https://github.com/fizyr/keras-retinanet
Francois Chollet, Keras code and weights files for popular deeplearning models https://github.com/fchollet/deep-learning-models
Forrest N. et al, SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size https://arxiv.org/abs/1602.07360
Kaiming H. et al, Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
Szegedy. et al, Rethinking the Inception Architecture for Computer Vision https://arxiv.org/abs/1512.00567
Gao. et al, Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993
Tsung-Yi. et al, Focal Loss for Dense Object Detection https://arxiv.org/abs/1708.02002
O Russakovsky et al, ImageNet Large Scale Visual Recognition Challenge https://arxiv.org/abs/1409.0575
TY Lin et al, Microsoft COCO: Common Objects in Context https://arxiv.org/abs/1405.0312
Moses & John Olafenwa, A collection of images of identifiable professionals. https://github.com/OlafenwaMoses/IdenProf
Joseph Redmon and Ali Farhadi, YOLOv3: An Incremental Improvement. https://arxiv.org/abs/1804.02767
Experiencor, Training and Detecting Objects with YOLO3 https://github.com/experiencor/keras-yolo3
MobileNetV2: Inverted Residuals and Linear Bottlenecks https://arxiv.org/abs/1801.04381
YOLOv3 in PyTorch > ONNX > CoreML > TFLite https://github.com/ultralytics/yolov3
