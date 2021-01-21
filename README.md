# Facial key-points

### Adalberto Gonzalez

Computer Vision Nanodegree Program

Objective: In this project, I've combined computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face!

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition.  Some examples of these keypoints are pictured below.

<img src="https://video.udacity-data.com/topher/2018/April/5acd7ff3_screen-shot-2018-04-10-at-8.24.14-pm/screen-shot-2018-04-10-at-8.24.14-pm.png" alt="img" width="400" />



------

## Project Structure

The project is broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) were graded**:

**Notebook 1** : Loading and Visualizing the Facial Keypoint Data

**Notebook 2** : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

**Notebook 3** : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

**Notebook 4** : Fun Filters and Keypoint Uses

**models.py**

------

## **models.py**

I  configured a functional convolutional neural network along with the feedforward behavior

#### 

```python
self.conv1 = nn.Conv2d(1, 32, 5)#32x220x220
self.pool1 = nn.MaxPool2d(2, 2) #32x110x110
self.batch1 = nn.BatchNorm2d(32) #106x106

self.conv2 = nn.Conv2d(32, 64, 5) #64x106x106
self.pool1 = nn.MaxPool2d(2, 2)    #64x53x53
self.batch2 = nn.BatchNorm2d(64)  #64x49x49

self.conv3 = nn.Conv2d(64, 128, 5) #128x49x49
self.pool1 = nn.MaxPool2d(2, 2)    #128x24x24
self.batch3 = nn.BatchNorm2d(128) #128x22x22

self.conv4 = nn.Conv2d(128, 256, 5) #256x22x22
self.pool1 = nn.MaxPool2d(2, 2)     #256x11x11
self.batch4 = nn.BatchNorm2d(256)  #256x11x11

## added, maxpooling layers, multiple conv layers, fully-connected layers also dropout and batch normalization to avoid overfitting.
## Max pooling is a simple down-sampling operation and I am sure you know what it does. nn.MaxPool2d returns a stateless (has no trainable parameters) object so you do not have to declare multiple instances with the same kernel size, you can just reuse the same object multiple times. :wink:
   
self.fc1 = nn.Linear(256*10*10, 1000) ### ****
self.fc1_drop = nn.Dropout(p=0.2)
self.fc2 = nn.Linear(1000, 136)
```

#### 

## Notebook 2: Define the Network Architecture

I used the provided transforms inside `data_transform` to turn an input image into a normalized, square, grayscale image in Tensor format. You have also nicely added *RandomCrop* operation to perform Data Augmentation. Well done! :smile:

```python
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

```

### 

I selected the folowing loss function and optimizer:Adam` is a great (also a [safe](https://arxiv.org/abs/2007.01547) ) choice for an optimizer. `SmoothL1Loss` is a very reliable alternative to `MSELoss`, as it tends to be more robust to outliers in the data while training. It combines the advantages of both L1-loss (steady gradients for large values of x) and L2-loss (fewer oscillations during updates when x is small). Good choice! ! :blush:

```python
import torch.optim as optim

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

```



The model has trained well considering the following: I started with epoc=2 then I increased the batch to 64 to review if this could give me a better performance, and it worked. Once I saw the have validation loss accuracy going in the right direction I increased the epoch 20,30 and lastly 50. one Issue that i didnt expected was in the virtual environment, the version we have to use has an unsolved bug wich obliged me to use 0 workers.  I could have used 4 and then increase epochs to 100 al least, however I had to make a choice based on time availabiliy.

```pyhton
Epoch: 50, Batch: 30, Avg. Loss: 0.008666319865733385
Epoch: 50, Batch: 40, Avg. Loss: 0.015493830433115363
Epoch: 50, Batch: 50, Avg. Loss: 0.008042958611622453
Finished Training
```

It is never a bad idea to see and compare training and validation losses. You can visualize them with this piece of code:

```python
loss_data = np.asarray(loss_data)
plt.plot(loss_data[:,0], label='Training Loss')
plt.plot(loss_data[:,1], label='Validation Loss')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_alpha(0.7)
ax.spines['bottom'].set_alpha(0.7)
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
```



## Notebook 3: Facial keypoint Detection.

Add padding to the faces is a very effective trick. Looking at the training images in Notebook 2, I bet you would agree with me that the faces in the dataset are not as zoomed in as the ones Haar Cascade detects. This is why you MUST grab more area around the detected faces to make sure the entire head (the curvature) is present in the input image. You can do the padding in a generic way, without having to use a constant padding value, with the following code:

```python
margin = int(w*0.3)
roi = image_copy[max(y-margin,0):min(y+h+margin,image.shape[0]), 
                 max(x-margin,0):min(x+w+margin,image.shape[1])]
```

------

The model has been applied and the predicted key-points are being displayed on each face in the image. This model predictions look acceptable. 


<img src="https://github.com/Adal-Glez/ComputerVision_ND_P1_Facial_Keypoints/blob/master/model_results/01.png"/> 
<img src="https://github.com/Adal-Glez/ComputerVision_ND_P1_Facial_Keypoints/blob/master/model_results/02.png"/>

Thanks for reading.

**Adalberto**

