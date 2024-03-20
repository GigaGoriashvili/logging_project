import numpy as np
from typing import List
import logging.config
import yaml

# Task 4
# Configuring using YAML file and !!creating logging infrastructure
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.config.dictConfig(config)


class LS:
    def __init__(self, data: List):
        min_x, column_headers = LS.__apply_data(data)
        self.__min_x = min_x  # Store the solution vector
        self.__column_headers = column_headers

        self.__logger = logging.getLogger('LS')  # creating logging infrastructure

        self.__logger.info('New instance of LS created')

    @staticmethod
    def __projection(v, u):
        # Compute the projection of vector v onto vector u
        return (np.dot(v, u) / np.dot(u, u)) * u

    @staticmethod
    def __gram_schmidt(matrix):
        # Implement the classical Gram-Schmidt orthogonalization process
        row_num, col_num = matrix.shape
        orthogonal_matrix = np.zeros((row_num, col_num))
        orthogonal_matrix[:, 0] = matrix[:, 0]
        # Iterate through each column and orthogonalize
        for k in range(1, col_num):
            v_k = matrix[:, k]
            u_k = v_k
            for j in range(k):
                u_k = u_k - LS.__projection(v_k, orthogonal_matrix[:, j])
            orthogonal_matrix[:, k] = u_k
        # Normalize the orthogonal vectors
        for k in range(col_num):
            orthogonal_matrix[:, k] /= np.linalg.norm(orthogonal_matrix[:, k])
        return orthogonal_matrix

    @staticmethod
    def __qr(matrix):
        # Compute the QR decomposition using Gram-Schmidt process
        Q = LS.__gram_schmidt(matrix)
        R = np.dot(np.transpose(Q), matrix)
        return Q, R

    @staticmethod
    def __apply_data(data):
        # Prepare data for computation
        column_headers = data[0]
        parameters = [row[:-1] for row in data[1:]]
        for row in parameters:
            row.append(1)
        result_b = [row[-1] for row in data[1:]]
        A_matrix = np.array(parameters)

        # Compute the solution using classical Gram-Schmidt for QR
        Q, R = LS.__qr(A_matrix)

        # Compute the solution using QR decomposition
        QTb = np.dot(np.transpose(Q), result_b)
        R_inv = np.linalg.inv(R)
        min_x = np.dot(R_inv, QTb)

        return min_x, column_headers

    def predict(self, input_data=None):
        try:
            # Predict the output based on the stored solution vector and input data
            if input_data is None:
                # If no input data is provided, prompt the user for input
                input_data = []
                for ingredient in self.__column_headers[:-1]:
                    num = float(input(f'Input parameter value of {ingredient}:'))
                    input_data.append(num)
                    self.__logger.info(f'Input value for {ingredient} provided by user: {num}')
            else:
                # Check if the provided input data has the correct length
                if len(input_data) != len(self.__min_x[:-1]):
                    raise ValueError('Invalid input!')
            # Compute the predicted output
            self.__logger.info('Input data provided: %s', input_data)
            predicted_num = sum(item1 * item2 for item1, item2 in zip(input_data, self.__min_x[:-1])) + self.__min_x[-1]
            self.__logger.info('Prediction successful: %s', predicted_num)
            return predicted_num
        except Exception as e:
            self.__logger.error('An error occurred: %s', e)
            raise
