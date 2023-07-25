{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "82891613",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.preprocessing import image\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "import cv2\n",
    "from keras.preprocessing.image import img_to_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c903bb3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(rescale = 1./255,\n",
    "shear_range = 0.2,\n",
    "zoom_range = 0.2,\n",
    "horizontal_flip = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "954e88a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images = r\"C:\\Users\\ROSHAAN\\Desktop\\Btumor\\dataset\\training_set\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "59df7c59",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 195 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_generator = train_datagen.flow_from_directory(train_images,\n",
    "    target_size = (300,300),\n",
    "    batch_size = 128,\n",
    "    class_mode = 'binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e5dac121",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'no': 0, 'yes': 1}"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_generator.class_indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "44ec67f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 58 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "test_datagen = ImageDataGenerator(rescale = 1./255)\n",
    "validation_generator = test_datagen.flow_from_directory(r'C:\\Users\\ROSHAAN\\Desktop\\Btumor\\dataset\\test_set',\n",
    "    target_size= (300,300),\n",
    "    batch_size = 128,\n",
    "    class_mode = 'binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8b9c66a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 298, 298, 16)      448       \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2D  (None, 149, 149, 16)     0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " conv2d_1 (Conv2D)           (None, 147, 147, 32)      4640      \n",
      "                                                                 \n",
      " max_pooling2d_1 (MaxPooling  (None, 73, 73, 32)       0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " conv2d_2 (Conv2D)           (None, 71, 71, 64)        18496     \n",
      "                                                                 \n",
      " max_pooling2d_2 (MaxPooling  (None, 35, 35, 64)       0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " conv2d_3 (Conv2D)           (None, 33, 33, 128)       73856     \n",
      "                                                                 \n",
      " max_pooling2d_3 (MaxPooling  (None, 16, 16, 128)      0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " conv2d_4 (Conv2D)           (None, 14, 14, 128)       147584    \n",
      "                                                                 \n",
      " max_pooling2d_4 (MaxPooling  (None, 7, 7, 128)        0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 6272)              0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 256)               1605888   \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 512)               131584    \n",
      "                                                                 \n",
      " dense_2 (Dense)             (None, 1)                 513       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 1,983,009\n",
      "Trainable params: 1,983,009\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model= tf.keras.models.Sequential([\n",
    "                                   tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape= (300, 300, 3)),\n",
    "                                   tf.keras.layers.MaxPool2D(2,2),\n",
    "                                   tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),\n",
    "                                   tf.keras.layers.MaxPool2D(2,2),\n",
    "                                   tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),\n",
    "                                   tf.keras.layers.MaxPool2D(2,2),\n",
    "                                   tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),\n",
    "                                   tf.keras.layers.MaxPool2D(2,2),\n",
    "                                   tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),\n",
    "                                   tf.keras.layers.MaxPool2D(2,2),\n",
    "\n",
    "                                   tf.keras.layers.Flatten(),\n",
    "                                   tf.keras.layers.Dense(256, activation= 'relu'),\n",
    "                                   tf.keras.layers.Dense(512, activation= 'relu'),\n",
    "                                   tf.keras.layers.Dense(1, activation= 'sigmoid')\n",
    "])\n",
    "model.summary()\n",
    "model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0625dd43",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "2/2 [==============================] - 16s 5s/step - loss: 0.6975 - accuracy: 0.4205 - val_loss: 0.6815 - val_accuracy: 0.7241\n",
      "Epoch 2/50\n",
      "2/2 [==============================] - 16s 5s/step - loss: 0.6726 - accuracy: 0.6974 - val_loss: 0.6841 - val_accuracy: 0.5000\n",
      "Epoch 3/50\n",
      "2/2 [==============================] - 15s 4s/step - loss: 0.6324 - accuracy: 0.6462 - val_loss: 0.6528 - val_accuracy: 0.5517\n",
      "Epoch 4/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.5732 - accuracy: 0.7128 - val_loss: 0.6341 - val_accuracy: 0.6034\n",
      "Epoch 5/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.6034 - accuracy: 0.6872 - val_loss: 0.5662 - val_accuracy: 0.7069\n",
      "Epoch 6/50\n",
      "2/2 [==============================] - 13s 9s/step - loss: 0.5518 - accuracy: 0.7282 - val_loss: 0.4894 - val_accuracy: 0.8448\n",
      "Epoch 7/50\n",
      "2/2 [==============================] - 15s 4s/step - loss: 0.5643 - accuracy: 0.6974 - val_loss: 0.5046 - val_accuracy: 0.8793\n",
      "Epoch 8/50\n",
      "2/2 [==============================] - 15s 10s/step - loss: 0.5589 - accuracy: 0.7179 - val_loss: 0.5077 - val_accuracy: 0.8448\n",
      "Epoch 9/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.5583 - accuracy: 0.7231 - val_loss: 0.5486 - val_accuracy: 0.7414\n",
      "Epoch 10/50\n",
      "2/2 [==============================] - 15s 10s/step - loss: 0.5429 - accuracy: 0.7333 - val_loss: 0.4821 - val_accuracy: 0.8276\n",
      "Epoch 11/50\n",
      "2/2 [==============================] - 15s 4s/step - loss: 0.5049 - accuracy: 0.7692 - val_loss: 0.4487 - val_accuracy: 0.8276\n",
      "Epoch 12/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.5351 - accuracy: 0.7333 - val_loss: 0.4469 - val_accuracy: 0.8103\n",
      "Epoch 13/50\n",
      "2/2 [==============================] - 15s 9s/step - loss: 0.4856 - accuracy: 0.7744 - val_loss: 0.4646 - val_accuracy: 0.7759\n",
      "Epoch 14/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.4776 - accuracy: 0.8000 - val_loss: 0.4906 - val_accuracy: 0.7759\n",
      "Epoch 15/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.4320 - accuracy: 0.8000 - val_loss: 0.5217 - val_accuracy: 0.7759\n",
      "Epoch 16/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.4477 - accuracy: 0.8256 - val_loss: 0.6302 - val_accuracy: 0.6724\n",
      "Epoch 17/50\n",
      "2/2 [==============================] - 15s 10s/step - loss: 0.4898 - accuracy: 0.7641 - val_loss: 0.6156 - val_accuracy: 0.6897\n",
      "Epoch 18/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.4424 - accuracy: 0.7641 - val_loss: 0.6027 - val_accuracy: 0.7241\n",
      "Epoch 19/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.4354 - accuracy: 0.8154 - val_loss: 0.6143 - val_accuracy: 0.6552\n",
      "Epoch 20/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.4309 - accuracy: 0.7744 - val_loss: 0.5466 - val_accuracy: 0.7414\n",
      "Epoch 21/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.3927 - accuracy: 0.8154 - val_loss: 0.5232 - val_accuracy: 0.7931\n",
      "Epoch 22/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.4036 - accuracy: 0.8256 - val_loss: 0.5246 - val_accuracy: 0.7759\n",
      "Epoch 23/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.3943 - accuracy: 0.8103 - val_loss: 0.5522 - val_accuracy: 0.7931\n",
      "Epoch 24/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.3500 - accuracy: 0.8513 - val_loss: 0.5638 - val_accuracy: 0.7931\n",
      "Epoch 25/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.3387 - accuracy: 0.8615 - val_loss: 0.5676 - val_accuracy: 0.7759\n",
      "Epoch 26/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.3681 - accuracy: 0.8410 - val_loss: 0.5822 - val_accuracy: 0.7759\n",
      "Epoch 27/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.3481 - accuracy: 0.8410 - val_loss: 0.5953 - val_accuracy: 0.7759\n",
      "Epoch 28/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.3283 - accuracy: 0.8615 - val_loss: 0.6332 - val_accuracy: 0.7414\n",
      "Epoch 29/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.3614 - accuracy: 0.8359 - val_loss: 0.6009 - val_accuracy: 0.7759\n",
      "Epoch 30/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.3133 - accuracy: 0.8564 - val_loss: 0.6128 - val_accuracy: 0.7759\n",
      "Epoch 31/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2886 - accuracy: 0.8821 - val_loss: 0.5701 - val_accuracy: 0.7586\n",
      "Epoch 32/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.2773 - accuracy: 0.8667 - val_loss: 0.6471 - val_accuracy: 0.7414\n",
      "Epoch 33/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2945 - accuracy: 0.8769 - val_loss: 0.6892 - val_accuracy: 0.7759\n",
      "Epoch 34/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.2807 - accuracy: 0.8667 - val_loss: 0.7028 - val_accuracy: 0.7241\n",
      "Epoch 35/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.3348 - accuracy: 0.8718 - val_loss: 0.7234 - val_accuracy: 0.7414\n",
      "Epoch 36/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.2689 - accuracy: 0.8974 - val_loss: 0.6892 - val_accuracy: 0.7414\n",
      "Epoch 37/50\n",
      "2/2 [==============================] - 13s 4s/step - loss: 0.2775 - accuracy: 0.9128 - val_loss: 0.6696 - val_accuracy: 0.7759\n",
      "Epoch 38/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2575 - accuracy: 0.9231 - val_loss: 0.6569 - val_accuracy: 0.7931\n",
      "Epoch 39/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2985 - accuracy: 0.8410 - val_loss: 0.7631 - val_accuracy: 0.7241\n",
      "Epoch 40/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.2100 - accuracy: 0.9385 - val_loss: 0.8253 - val_accuracy: 0.7414\n",
      "Epoch 41/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2067 - accuracy: 0.9385 - val_loss: 0.8246 - val_accuracy: 0.7414\n",
      "Epoch 42/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2317 - accuracy: 0.8923 - val_loss: 0.6610 - val_accuracy: 0.8103\n",
      "Epoch 43/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2235 - accuracy: 0.9077 - val_loss: 0.6266 - val_accuracy: 0.8448\n",
      "Epoch 44/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.2962 - accuracy: 0.8667 - val_loss: 0.7632 - val_accuracy: 0.7586\n",
      "Epoch 45/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.1722 - accuracy: 0.9333 - val_loss: 0.8180 - val_accuracy: 0.7414\n",
      "Epoch 46/50\n",
      "2/2 [==============================] - 14s 9s/step - loss: 0.2737 - accuracy: 0.8821 - val_loss: 0.8160 - val_accuracy: 0.6897\n",
      "Epoch 47/50\n",
      "2/2 [==============================] - 15s 4s/step - loss: 0.2205 - accuracy: 0.9077 - val_loss: 0.6132 - val_accuracy: 0.8103\n",
      "Epoch 48/50\n",
      "2/2 [==============================] - 15s 4s/step - loss: 0.2313 - accuracy: 0.9026 - val_loss: 0.5843 - val_accuracy: 0.8448\n",
      "Epoch 49/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.1596 - accuracy: 0.9436 - val_loss: 0.6825 - val_accuracy: 0.7931\n",
      "Epoch 50/50\n",
      "2/2 [==============================] - 14s 4s/step - loss: 0.1821 - accuracy: 0.9436 - val_loss: 0.8328 - val_accuracy: 0.8103\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(train_generator, epochs = 50, validation_data = validation_generator)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8237911e",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"trained.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "8b121161",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import load_model\n",
    "model = load_model(\"trained.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "4811b8c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.99924856]], dtype=float32)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "img= cv2.imread(r\"C:\\Users\\ROSHAAN\\Desktop\\Btumor\\dataset\\test_set\\yes\\Y183.jpg\")\n",
    "tempimg = img\n",
    "img = cv2.resize(img,(300,300))\n",
    "img = img/255.0\n",
    "img = img.reshape(1,300,300,3)\n",
    "model.predict(img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "d4934811",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: Brain Tumor\n"
     ]
    }
   ],
   "source": [
    "prediction = model.predict(img) >= 0.5\n",
    "if prediction>=0.5:\n",
    "  prediction = \"Brain Tumor\"\n",
    "else:\n",
    "  prediction = \"Normal\"\n",
    "print(\"Prediction: \"+prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "978b37b9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
