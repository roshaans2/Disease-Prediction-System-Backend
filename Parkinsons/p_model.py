{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "502c4e69",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn import svm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f3cd35cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "parkinsons_data = pd.read_csv(\"parkinsons.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "467c7704",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = parkinsons_data.drop(columns=['name','status'],axis=1)\n",
    "Y = parkinsons_data['status'] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b9bdc2bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f92e7b65",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c9a7f34e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "StandardScaler()"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaler.fit(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d2392098",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = scaler.transform(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f0ba2e9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1ca7f7f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = svm.SVC(kernel='linear')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9d5e2150",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVC(kernel='linear')"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "34aa12d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Python310\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# input_data = (162.56800,198.34600,77.63000,0.00502,0.00003,0.00280,0.00253,0.00841,0.01791,0.16800,0.00793,0.01057,0.01799,0.02380,0.01170,25.67800,0.427785,0.723797,-6.635729,0.209866,1.957961,0.135242)\n",
    "\n",
    "input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)\n",
    "\n",
    "\n",
    "input_data_as_numpy_array = np.asarray(input_data)\n",
    "\n",
    "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
    "\n",
    "std_data = scaler.transform(input_data_reshaped)\n",
    "\n",
    "prediction = model.predict(std_data)\n",
    "\n",
    "print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b43b179c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6e182b5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['parkinsons_model']"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(model,\"parkinsons_model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aaea14ba",
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
