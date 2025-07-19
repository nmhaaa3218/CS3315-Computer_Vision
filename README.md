# Computer Vision Assignments Portfolio
**Student:** Manh Ha Nguyen (a1840406)  
**Course:** CS3315 - Computer Vision  
**University:** University of Auckland  

This repository contains three comprehensive assignments covering fundamental to advanced computer vision concepts, from basic image processing to deep learning applications.

## 📋 Table of Contents
- [Assignment 1: Image Processing Fundamentals](#assignment-1-image-processing-fundamentals)
- [Assignment 2: Interactive Computer Graphics](#assignment-2-interactive-computer-graphics)
- [Assignment 3: Deep Learning for Perception](#assignment-3-deep-learning-for-perception)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)

---

## 🖼️ Assignment 1: Image Processing Fundamentals

### Overview
A comprehensive introduction to basic image processing operations, implementing core algorithms from scratch using NumPy and scikit-image.

### Key Features
- **Image Loading & Display**: Implemented image loading with normalization (0-1 range)
- **Basic Operations**: Crop, resize, contrast adjustment, and color space conversion
- **Filtering**: 2D convolution implementation for both grayscale and RGB images
- **Advanced Processing**: Gaussian filtering, Sobel edge detection, and Laplacian of Gaussian (LoG) blob detection
- **Image Pyramids**: Multi-scale image representation using Gaussian pyramids

### Core Functions Implemented
```python
# Basic Operations
load(img_path)           # Load and normalize images
crop(image, start_row, start_col, num_rows, num_cols)  # Image cropping
resize(input_image, output_rows, output_cols)          # Nearest neighbor resizing
change_contrast(image, factor)                         # Contrast adjustment
greyscale(input_image)                                 # RGB to grayscale conversion
binary(grey_img, threshold)                            # Binary thresholding

# Advanced Processing
conv2D(image, kernel)                                  # 2D convolution
conv(image, kernel)                                    # Multi-channel convolution
gauss2D(size, sigma)                                   # Gaussian kernel generation
LoG2D(size, sigma)                                     # Laplacian of Gaussian filter
```

### Tasks Completed
1. **Task 1**: Image loading and point processing operations
2. **Task 2**: Basic image processing functions (crop, resize, contrast, etc.)
3. **Task 3**: 2D convolution and filtering (Gaussian, Sobel edge detection)
4. **Task 4**: Image sampling and Gaussian pyramids
5. **Task 5**: Blob detection using LoG filters

### Files
- `Assignment_1/a1code.py` - Core implementation
- `Assignment_1/Assignment_1_Notebook.ipynb` - Jupyter notebook with experiments
- `Assignment_1/images/` - Test images directory

---

## ⏰ Assignment 2: Interactive Computer Graphics

### Overview
An interactive p5.js application demonstrating real-time computer graphics with time-based animations and user interaction.

### Key Features
- **Dynamic Day/Night Cycle**: Realistic sky color transitions based on simulated time
- **Interactive Clock**: Frameless analog clock with day/night color adaptation
- **Celestial Animation**: Sun/moon movement along a curved arc
- **Star Field**: Dynamic star rendering during night hours
- **Time Control**: User-controlled time acceleration with multiple speed levels
- **Smooth Transitions**: Color interpolation for seamless day/night transitions

### Technical Implementation
```javascript
// Core Animation Functions
drawBackground()    // Renders dynamic sky and ground
drawClock()         // Renders interactive analog clock
getAlpha(timeInHours) // Calculates day/night blend factor
speedUp()           // User interaction for time acceleration
```

### Visual Elements
- **Sky Gradient**: Smooth transition from day (light yellow) to night (dark blue)
- **Ground Arc**: Curved horizon with matching day/night colors
- **Celestial Body**: Sun/moon that moves along the horizon arc
- **Star Field**: 100 randomly positioned stars visible at night
- **Clock Face**: Traditional analog clock with adaptive colors

### Interactive Features
- **Speed Control**: Button to accelerate time simulation
- **Real-time Updates**: 60 FPS animation with smooth transitions
- **Responsive Design**: Adapts to canvas size changes

### Files
- `Assignment_2/code.txt` - Main p5.js implementation

---

## 🧠 Assignment 3: Deep Learning for Perception

### Overview
Advanced deep learning implementation using PyTorch for image classification on the Fashion-MNIST dataset, exploring neural network architectures and training dynamics.

### Key Features
- **Multi-Architecture Networks**: Implemented both fully connected and CNN architectures
- **Comprehensive Training**: Learning rate experiments and convergence analysis
- **Performance Analysis**: Detailed accuracy and loss tracking
- **Weight Initialization**: Xavier initialization and activation function experiments
- **Practical Applications**: Research proposal on computer vision applications

### Network Architectures

#### Fully Connected Network
```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
```

#### Convolutional Neural Network
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # ... fully connected layers
```

### Experiments Conducted
1. **Learning Rate Analysis**: Testing 0.001, 0.01, 0.1, and 1.0 learning rates
2. **Convergence Studies**: Training for up to 100 epochs to observe convergence
3. **Architecture Comparison**: FC vs CNN performance analysis
4. **Activation Functions**: ReLU, Tanh, and Sigmoid comparison
5. **Weight Initialization**: Xavier vs random initialization effects

### Results Summary
| Learning Rate | Best Accuracy | Convergence Epoch |
|---------------|---------------|-------------------|
| 1.0           | 19.97%        | 1                 |
| 0.1           | 87.53%        | 13                |
| 0.01          | 83.56%        | 30                |
| 0.001         | 71.35%        | 30                |

### Files
- `Assignment_3/Assignment3_2025.ipynb` - Main Jupyter notebook
- `Assignment_3/Assignment3_2025.html` - HTML export
- `Assignment_3/Assignment3_2025_updated.html` - Updated version

---

## 🛠️ Technologies Used

### Programming Languages
- **Python 3.x** - Primary language for image processing and deep learning
- **JavaScript (p5.js)** - Interactive graphics and animations
- **HTML/CSS** - Web interface and documentation

### Libraries & Frameworks
- **NumPy** - Numerical computing and array operations
- **scikit-image** - Image processing utilities
- **PyTorch** - Deep learning framework
- **Matplotlib** - Data visualization and plotting
- **p5.js** - Creative coding and interactive graphics
- **Jupyter Notebook** - Interactive development and documentation

### Key Dependencies
```
numpy>=1.21.0
scikit-image>=0.18.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.4.0
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Node.js (for p5.js development)
- Jupyter Notebook or JupyterLab

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CS3315-Computer_Vision
   ```

2. **Install Python dependencies**
   ```bash
   pip install numpy scikit-image torch torchvision matplotlib jupyter
   ```

3. **For Assignment 2 (p5.js)**
   - Open `Assignment_2/code.txt` in a p5.js editor
   - Or use the p5.js web editor: https://editor.p5js.org/

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## 🚀 Usage

### Assignment 1: Image Processing
```python
# Load and process images
from Assignment_1.a1code import *

# Load an image
img = load('path/to/image.jpg')

# Apply various operations
cropped = crop(img, 100, 100, 200, 200)
resized = resize(img, 512, 512)
filtered = conv(img, gauss2D(15, 2))
```

### Assignment 2: Interactive Graphics
- Open the p5.js code in a web editor
- Click "Speed Up" to accelerate time
- Observe the dynamic day/night cycle and clock animation

### Assignment 3: Deep Learning
```python
# Run the Jupyter notebook
jupyter notebook Assignment_3/Assignment3_2025.ipynb

# Or execute specific cells for experiments
# - Learning rate experiments
# - Architecture comparisons
# - Training and evaluation
```

---

## 📊 Learning Outcomes

### Technical Skills Developed
- **Image Processing**: Understanding of fundamental algorithms and their implementation
- **Computer Graphics**: Real-time rendering and interactive applications
- **Deep Learning**: Neural network design, training, and evaluation
- **Software Engineering**: Code organization, testing, and documentation

### Key Concepts Mastered
- **Convolution Operations**: 2D filtering and feature extraction
- **Multi-scale Processing**: Image pyramids and scale-space analysis
- **Neural Network Training**: Loss functions, optimizers, and convergence
- **Interactive Systems**: User input handling and real-time updates

### Research Skills
- **Experimental Design**: Systematic testing of different parameters
- **Data Analysis**: Performance evaluation and result interpretation
- **Technical Writing**: Clear documentation and result presentation

---

## 📝 Notes

- All assignments include comprehensive error handling and input validation
- Code follows Python and JavaScript best practices
- Extensive testing and validation performed on all implementations
- Results and analysis documented in Jupyter notebooks

---

**Author:** Manh Ha Nguyen  
**Student ID:** a1840406  
**Course:** CS3315 Computer Vision  
**University:** University of Auckland  
**Year:** 2025 