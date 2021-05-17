# View the image

img = vqa_train_df['img_path'][285]
print(img)
img = cv2.imread(img)
# img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
print("Shape of the image : ", img.shape)
# Since the image is being read in BGR we need to convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plotting the image
fig = plt.figure(figsize = (10, 6))
plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()


# Paths to the images
TRAIN_SOURCE_IMAGES = os.path.join("images/train2014/")
TEST_SOURCE_IMAGES = os.path.join("images/val2014/")

train_destination_path = os.path.join("data/train_images.h5")
test_destination_path = os.path.join("data/test_images.h5")

# Process the images
# Reference : http://docs.h5py.org/en/stable/high/dataset.html

def process_images(source_images_path, destination_path):
    """
    Saves compressed, resized images as HDF5 datsets
    """
    
    # Path of image
    SOURCE_IMAGES = source_images_path
    # Finding total images
    images = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))
    
    # Size of data
    NUM_IMAGES = len(images)
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    
    with h5py.File(destination_path, 'w') as hf: 
        for img in tqdm(images):
            
            # Reading images and reshaping them
            image = cv2.imread(img)
            image = cv2.resize(image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            dataset = hf.create_dataset(name=os.path.basename(img).split('.')[0],
                                        data=image,
                                        shape=(HEIGHT, WIDTH, CHANNELS),
                                        maxshape=(HEIGHT, WIDTH, CHANNELS),
                                        compression="gzip",
                                        compression_opts=9)
    print("DONE...")
