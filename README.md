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



pip install cython pillow>=7.0.0 

numpy>=1.18.1 opencv-python>=4.1.2 

torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cu102 

torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cu102 

pytest==7.1.3 

tqdm==4.64.1 

scipy>=1.7.3 

matplotlib>=3.4.3 

mock==4.0.3

If you plan to train custom AI models, download requirements_extra.txt file and install via the command

pip install imageai --upgrade
Features
Image Classification



Documentation - English Version https://imageai.readthedocs.io
Sponsors
Real-Time and High Performance Implementation
ImageAI provides abstracted and convenient implementations of state-of-the-art Computer Vision technologies. All of ImageAI implementations and code can work on any computer system with moderate CPU capacity. However, the speed of processing for operations like image prediction, object detection and others on CPU is slow and not suitable for real-time applications. To perform real-time Computer Vision operations with high performance, you need to use GPU enabled technologies.

ImageAI uses the PyTorch backbone for it's Computer Vision operations. PyTorch supports both CPUs and GPUs ( Specifically NVIDIA GPUs. You can get one for your PC or get a PC that has one) for machine learning and artificial intelligence algorithms' implementations.

Projects Built on ImageAI
AI Practice Recommendations
For anyone interested in building AI systems and using them for business, economic, social and research purposes, it is critical that the person knows the likely positive, negative and unprecedented impacts the use of such technologies will have. They must also be aware of approaches and practices recommended by experienced industry experts to ensure every use of AI brings overall benefit to mankind. We therefore recommend to everyone that wishes to use ImageAI and other AI tools and resources to read Microsoft's January 2018 publication on AI titled "The Future Computed : Artificial Intelligence and its role in society". Kindly follow the link below to download the publication.
