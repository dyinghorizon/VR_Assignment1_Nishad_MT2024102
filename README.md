# VR_Assignment1_Nishad_MT2024102

# Coin Detection and Segmentation

## Assignment Overview
This assignment implements computer vision techniques to detect, segment, and count Indian coins from an image. The implementation uses edge detection, thresholding, and contour analysis to accurately identify and count coins in a given image.

## Features
- Preprocessing with Gaussian filters at different sigma values
- Edge detection using both Sobel and Canny methods
- Image segmentation using thresholding techniques
- Contour detection and filtering
- Coin counting with visual representation

## Dependencies
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image

## Methodology

### 1. Image Preprocessing
The image is preprocessed using multiple Gaussian filters with different sigma values:
- `sigma=2` for edge detection : Moderated denoising id prefered for edge detection.
- `sigma=3` for general denoising
- `sigma=25` for segmentation : Less noise is preferable for coin segmentation and counting the coins. This can be adjusted depending on the image.

```python
denoised_coins_for_edge_detection = gaussian(coins, sigma=2)
denoised_coins = gaussian(coins, sigma=3)
denoised_coins_for_segmentation = gaussian(coins, sigma=25)
```

### 2. Edge Detection
Two different edge detection techniques are implemented and compared:

#### Sobel Edge Detection
```python
edge_sobel = sobel(gray_coins)
```

#### Canny Edge Detection
```python
canny_edge = canny(gray_coins_for_edge_detection, sigma=0.002)
```

### 3. Segmentation
Binary segmentation is performed using thresholding:

```python
# Manual thresholding
binary_global = gray_coins > 0.65

# Otsu's method (automatic thresholding)
thresh = threshold_otsu(gray_coins_for_segmentation)
```

### 4. Contour Detection and Counting
Contours are detected from the binary image and filtered by size to identify individual coins:

```python
contours = measure.find_contours(binary_global, 0.8)
min_ring_size = 1200
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] > min_ring_size]
```

## Results

### Edge Detection Results
- Sobel edge detection highlights the boundaries of coins
- Canny edge detection provides more precise edges with better noise suppression

### Segmentation Results
The binary thresholded image clearly separates coins from the background, creating solid regions for each coin.

### Counting Results
The algorithm successfully detected and counted 7 coins in the image, with each coin distinctly identified and outlined in a different color.

## Usage

1. Load an image containing coins:
```python
filename = '/path/to/your/image.jpg'
coins = ski.io.imread(filename)
```

2. Run the processing pipeline:
```python
# Apply Gaussian filtering
denoised_coins = gaussian(coins, sigma=3)

# Convert to grayscale
gray_coins = color.rgb2gray(denoised_coins)

# Apply thresholding
binary_global = gray_coins > 0.65

# Find contours
contours = measure.find_contours(binary_global, 0.8)

# Filter contours by size
min_ring_size = 1200
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] > min_ring_size]

# Display results
show_image_contour(coins, dots_contours)
print("Number of Coins: {}".format(len(dots_contours)))
```

## Visualization Functions

The project includes several helper functions for visualizing results:

```python
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_comparison(original, filtered, title_filt):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 9), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_filt)
    ax2.axis('off')

def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('Contours')
    plt.axis('off')
```

## Conclusion

The methods that we are using for coin detection are not intelligent, so the code will not be inherently robust. Depending on the image, we can tweak some parameters in the code such as sigma (for blurring), manual threshold values, and size of contours. By making these adjustments, the code will work for other images as well, assuming there is minimal noise in the background.
