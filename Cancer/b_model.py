{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dc2e2941",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import sklearn.datasets\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a507c662",
   "metadata": {},
   "outputs": [],
   "source": [
    "breast_cancer_dataset = sklearn.datasets.load_breast_cancer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "922f7b64",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_frame = pd.DataFrame(breast_cancer_dataset.data,columns = breast_cancer_dataset.feature_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "da94c498",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_frame['label'] = breast_cancer_dataset.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c4b98862",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data_frame.drop(columns='label',axis=1)\n",
    "Y = data_frame['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "93421ba9",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e1a0463a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Python310\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LogisticRegression()\n",
    "model.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3d2440e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Python310\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)\n",
    "\n",
    "input_data_as_numpy_array = np.asarray(input_data)\n",
    "\n",
    "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
    "\n",
    "prediction = model.predict(input_data_reshaped)\n",
    "\n",
    "print(prediction)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "43ecce89",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['breast_cancer_model']"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "joblib.dump(model,\"breast_cancer_model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3451f63",
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
