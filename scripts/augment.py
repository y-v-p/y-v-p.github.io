from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import misc

# import the original image(s)
img = Image.open("28x28.png").convert('L')
image = np.array(img, dtype=np.uint8)
images = []
images.append(image)
images = [image]*10000

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
sometimes = lambda aug: iaa.Sometimes(0.55, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images

        sometimes(iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=255, # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),

        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.Dropout((0.005, 0.01), per_channel=0.5), # randomly remove up to 10% of the pixel
                ]),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05), mode=ia.ALL, cval=255)), # sometimes move parts of the image around
            ],
            random_order=True
        ),

        iaa.Sometimes(1.0, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5)), # add gaussian noise to images
    ],
    random_order=True
)

images_aug = seq.augment_images(images)

x = np.zeros(shape=(28*28))
for i, image_aug in enumerate(images_aug):
    # print image_aug.flatten()
    x = np.vstack([x, image_aug.flatten()])
    #misc.imsave("aug_images/image_%05d.png" % (i,), image_aug)

x = np.delete(x,0,0)

for i in range(20):
  misc.imsave('batches/mnist_batch_'+`i`+'.png', x[500*i:500*(i+1),:])
misc.imsave('batches/mnist_batch_'+`20`+'.png', x[9500:,:]) # test set
