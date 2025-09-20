import os
import cv2
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import random

def divideAndBalanceByClass(n_classes_balance, test_size):
    read_folder = "D:\\Old_PC\\doc\\Projects\\Elective_Study\\Dataset\\New_CRIC_Dataset_6"
    classes = os.listdir(read_folder)
    X_train_bal, y_train_bal, X_validation_bal, y_validation_bal, X_test_bal, y_test_bal = [], [], [], [], [], []
    
    x = 0 #ASCH
    read_folder2 = read_folder + '\\' + classes[x] + '\\'
    image_names = os.listdir(read_folder2)
    imgs_train, imgs_test = train_test_split(image_names, test_size=test_size)
    X_train, y_train, X_test, y_test = dataAugmentationASCH(imgs_train, imgs_test, n_classes_balance, read_folder2)
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    X_train_bal = X_train_bal + X_train_part
    y_train_bal = y_train_bal + y_train_part
    X_validation_bal = X_validation_bal + X_validation
    y_validation_bal = y_validation_bal + y_validation
    X_test_bal = X_test_bal + X_test
    y_test_bal = y_test_bal + y_test
    
    x = 1 #ASCUS
    read_folder2 = read_folder + '\\' + classes[x] + '\\'
    image_names = os.listdir(read_folder2)
    imgs_train, imgs_test = train_test_split(image_names, test_size=test_size)
    X_train, y_train, X_test, y_test = dataAugmentationASCUS(imgs_train, imgs_test, n_classes_balance, read_folder2)
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    X_train_bal = X_train_bal + X_train_part
    y_train_bal = y_train_bal + y_train_part
    X_validation_bal = X_validation_bal + X_validation
    y_validation_bal = y_validation_bal + y_validation
    X_test_bal = X_test_bal + X_test
    y_test_bal = y_test_bal + y_test
    
    x = 2 #CA
    read_folder2 = read_folder + '\\' + classes[x] + '\\'
    image_names = os.listdir(read_folder2)
    imgs_train, imgs_test = train_test_split(image_names, test_size=test_size)
    X_train, y_train, X_test, y_test = dataAugmentationCA(imgs_train, imgs_test, n_classes_balance, read_folder2)
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    X_train_bal = X_train_bal + X_train_part
    y_train_bal = y_train_bal + y_train_part
    X_validation_bal = X_validation_bal + X_validation
    y_validation_bal = y_validation_bal + y_validation
    X_test_bal = X_test_bal + X_test
    y_test_bal = y_test_bal + y_test
    
    x = 3 #HSIL
    read_folder2 = read_folder + '\\' + classes[x] + '\\'
    image_names = os.listdir(read_folder2)
    imgs_train, imgs_test = train_test_split(image_names, test_size=test_size)
    X_train, y_train, X_test, y_test = dataAugmentationHSIL(imgs_train, imgs_test, n_classes_balance, read_folder2)
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    X_train_bal = X_train_bal + X_train_part
    y_train_bal = y_train_bal + y_train_part
    X_validation_bal = X_validation_bal + X_validation
    y_validation_bal = y_validation_bal + y_validation
    X_test_bal = X_test_bal + X_test
    y_test_bal = y_test_bal + y_test
    
    x = 4 #LSIL
    read_folder2 = read_folder + '\\' + classes[x] + '\\'
    image_names = os.listdir(read_folder2)
    imgs_train, imgs_test = train_test_split(image_names, test_size=test_size)
    X_train, y_train, X_test, y_test = dataAugmentationLSIL(imgs_train, imgs_test, n_classes_balance, read_folder2)
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    X_train_bal = X_train_bal + X_train_part
    y_train_bal = y_train_bal + y_train_part
    X_validation_bal = X_validation_bal + X_validation
    y_validation_bal = y_validation_bal + y_validation
    X_test_bal = X_test_bal + X_test
    y_test_bal = y_test_bal + y_test
    
    x = 5 #Normal
    read_folder2 = read_folder + '\\' + classes[x] + '\\'
    image_names = os.listdir(read_folder2)
    imgs_train, imgs_test = train_test_split(image_names, test_size=test_size)
    X_train, y_train, X_test, y_test = dataAugmentationNormal(imgs_train, imgs_test, n_classes_balance, read_folder2)
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=test_size)
    X_train_bal = X_train_bal + X_train_part
    y_train_bal = y_train_bal + y_train_part
    X_validation_bal = X_validation_bal + X_validation
    y_validation_bal = y_validation_bal + y_validation
    X_test_bal = X_test_bal + X_test
    y_test_bal = y_test_bal + y_test
    
    return X_train_bal, y_train_bal, X_validation_bal, y_validation_bal, X_test_bal, y_test_bal



def dataAugmentationLSIL(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> 220 images become +1
    # For 3 classes -> each image becomes +1, 100 images become +1
    # For 2 classes -> 220 images become +1
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    if(n_classes == 6):
        class_id = 4 #LSIL
    elif(n_classes == 2):
        class_id = 0
    else:
        class_id = 1
    for x in range(0,n_images):
        img = cv2.imread(class_folder + training_images[x])
        X_training.append(img)
        y_training.append(class_id)
    if (n_classes == 3):
        image_operations = random.sample(range(0,n_images), n_images)
        for pos in range(0,len(image_operations)):
            if(pos < 100):
                operations = random.sample(range(1,10), 2)
            else:
                operations = random.sample(range(1,10), 1)
            for operation in operations:
                new_img = performAugmentationOperation(operation, X_training[pos])
                X_training.append(new_img)
                y_training.append(class_id)
    if (n_classes == 6 or n_classes == 2):
        image_operations = random.sample(range(0,n_images), 220)
        for img_op in image_operations:
            operation = random.randint(1,10)
            new_img = performAugmentationOperation(operation, X_training[img_op])
            X_training.append(new_img)
            y_training.append(class_id)
    for x in range(0,len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test


def dataAugmentationLSIL(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> 220 images become +1
    # For 3 classes -> each image becomes +1, 100 images become +1
    # For 2 classes -> 220 images become +1
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    if(n_classes == 6):
        class_id = 4 #LSIL
    elif(n_classes == 2):
        class_id = 0
    else:
        class_id = 1
    for x in range(0,n_images):
        img = cv2.imread(class_folder + training_images[x])
        X_training.append(img)
        y_training.append(class_id)
    if (n_classes == 3):
        image_operations = random.sample(range(0,n_images), n_images)
        for pos in range(0,len(image_operations)):
            if(pos < 100):
                operations = random.sample(range(1,10), 2)
            else:
                operations = random.sample(range(1,10), 1)
            for operation in operations:
                new_img = performAugmentationOperation(operation, X_training[pos])
                X_training.append(new_img)
                y_training.append(class_id)
    if (n_classes == 6 or n_classes == 2):
        image_operations = random.sample(range(0,n_images), 220)
        for img_op in image_operations:
            operation = random.randint(1,10)
            new_img = performAugmentationOperation(operation, X_training[img_op])
            X_training.append(new_img)
            y_training.append(class_id)
    for x in range(0,len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test

def dataAugmentationASCUS(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> each image becomes +1
    # For 3 classes -> each image becomes +3, 130 images become +1
    # For 2 classes -> each image becomes +1
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    if(n_classes == 6):
        class_id = 1 #ASCUS
    elif(n_classes == 2):
        class_id = 0
    else:
        class_id = 1
    for x in range(0, n_images):
        img = cv2.imread(class_folder + training_images[x])
        X_training.append(img)
        y_training.append(class_id)
    if (n_classes == 3):
        image_operations = random.sample(range(0, n_images), n_images)
        for pos in range(0, len(image_operations)):
            if(pos < 130):
                operations = random.sample(range(1, 10), 4)
            else:
                operations = random.sample(range(1, 10), 3)
            for operation in operations:
                new_img = performAugmentationOperation(operation, X_training[pos])
                X_training.append(new_img)
                y_training.append(class_id)
    if (n_classes == 6 or n_classes == 2):
        for x in range(0, n_images):
            operations = random.sample(range(1, 10), 2)
            for operation in operations:
                new_img = performAugmentationOperation(operation, X_training[x])
                X_training.append(new_img)
                y_training.append(class_id)
    for x in range(0, len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test

def dataAugmentationHSIL(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> no balancing
    # For 3 classes -> no balancing
    # For 2 classes -> no balancing
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    if(n_classes == 6):
        class_id = 3 #HSIL
    else:
        class_id = 0
    for x in range(0, n_images):
        img = cv2.imread(class_folder + training_images[x])
        X_training.append(img)
        y_training.append(class_id)
    for x in range(0, len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test

def dataAugmentationASCH(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> 270 images become +1
    # For 3 classes -> 270 images become +1
    # For 2 classes -> 270 images become +1
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    class_id = 0 #ASCH
    for x in range(0, n_images):
        img = cv2.imread(class_folder + training_images[x])
        X_training.append(img)
        y_training.append(class_id)
    image_operations = random.sample(range(0, n_images), 270)
    for img_op in image_operations:
        operation = random.randint(1, 10)
        new_img = performAugmentationOperation(operation, X_training[img_op])
        X_training.append(new_img)
        y_training.append(class_id)
    for x in range(0, len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test

def dataAugmentationCA(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> each CA image becomes +10
    # For 3 classes -> each CA image becomes +10
    # For 2 classes -> each CA image becomes +10
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    if(n_classes == 6):
        class_id = 2 #CA
    else:
        class_id = 0
    for x in range(0, n_images):
        img = cv2.imread(class_folder + training_images[x])
        X_training.append(img)
        y_training.append(class_id)
        for operation in range(1, 11):
            new_img = performAugmentationOperation(operation, img)
            X_training.append(new_img)
            y_training.append(class_id)
    for x in range(0, len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test
    

def dataAugmentationNormal(training_images, test_images, n_classes, class_folder):
    # For 6 classes -> already balanced
    # For 3 classes -> each normal image becomes +2
    # For 2 classes -> each image becomes +4
    n_images = len(training_images)
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    if(n_classes == 6):
        class_id = 5 #Normal
    elif(n_classes == 3):
        class_id = 2
    else:
        class_id = 1
    if(n_classes == 6):
    # No balancing needed
        for x in range(0, n_images):
            img = cv2.imread(class_folder + training_images[x])
            X_training.append(img)
            y_training.append(class_id)
    if(n_classes == 3):
    # Each image becomes +2
        for x in range(0, n_images):
            img = cv2.imread(class_folder + training_images[x])
            X_training.append(img)
            y_training.append(class_id)
            operations = random.sample(range(1, 10), 4)
            for operation in operations:
                new_img = performAugmentationOperation(operation, img)
                X_training.append(new_img)
                y_training.append(class_id)
    if(n_classes == 2):
    # Each image becomes +4
        for x in range(0, n_images):
            img = cv2.imread(class_folder + training_images[x])
            X_training.append(img)
            y_training.append(class_id)
            operations = random.sample(range(1, 10), 4)
            for operation in operations:
                new_img = performAugmentationOperation(operation, img)
                X_training.append(new_img)
                y_training.append(class_id)
    for x in range(0, len(test_images)):
        X_test.append(cv2.imread(class_folder + test_images[x]))
        y_test.append(class_id)
    return X_training, y_training, X_test, y_test



def performAugmentationOperation(operation, img):
    # rotate
    if(operation == 1):
        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif(operation == 2):
        new_img = cv2.rotate(img, cv2.ROTATE_180)
    elif(operation == 3):
        new_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # flip
    elif(operation == 4):
        new_img = cv2.flip(img, 1)
    elif(operation == 5):
        img_rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        new_img = cv2.flip(img_rotate_90, 1)
    elif(operation == 6):
        img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
        new_img = cv2.flip(img_rotate_180, 1)
    elif(operation == 7):
        img_rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_img = cv2.flip(img_rotate_270, 1)
    elif(operation == 8):
        sigma = 0.05 
        noisy = random_noise(img, var=sigma**2)
        new_img = noisy
        new_img = new_img * 255
    elif(operation == 9):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        new_img = denoise_tv_chambolle(noisy, weight=0.1, channel_axis=-1)

        new_img = new_img * 255
    elif(operation == 10):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        new_img = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)

        new_img = new_img * 255
    return new_img


def save_DividedBalanced(num_classes):
    X_train, y_train, X_validation, y_validation, X_test, y_test = divideAndBalanceByClass(num_classes, 0.2)
    save_folder = "D:\\Old_PC\\doc\\Projects\\Elective_Study\\Dataset\\6classes"
    for x in range(0, len(X_train)):
        new_name = save_folder + "\\Train\\" + str(x) + "_" + str(y_train[x]) + "_" + ".png"
        cv2.imwrite(new_name, X_train[x])
    for x in range(0, len(X_validation)):
        new_name = save_folder + "\\Validation\\" + str(x) + "_" + str(y_validation[x]) + "_" + ".png"
        cv2.imwrite(new_name, X_validation[x])
    for x in range(0, len(X_test)):
        new_name = save_folder + "\\Test\\" + str(x) + "_" + str(y_test[x]) + "_" + ".png"
        cv2.imwrite(new_name, X_test[x])
        
def read_DividedBalanced(num_classes):
    read_folder = "D:\\Old_PC\\doc\\Projects\\Elective_Study\\Dataset\\6classes"
    X_train, y_train, X_validation, y_validation, X_test, y_test = [], [], [], [], [], []
    for file in os.listdir(read_folder + '\\Train\\'):
        X_train.append(cv2.imread(read_folder + '\\Train\\' + file))
        y_train.append(file.split('_')[1])
    for file in os.listdir(read_folder + '\\Validation\\'):
        X_validation.append(cv2.imread(read_folder + '\\Validation\\' + file))
        y_validation.append(file.split('_')[1])
    for file in os.listdir(read_folder + '\\Test\\'):
        X_test.append(cv2.imread(read_folder + '\\Test\\' + file))
        y_test.append(file.split('_')[1])
    return X_train, y_train, X_validation, y_validation, X_test, y_test
