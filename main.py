import csv

import least_squares


def make_list(input_file):
    with open(input_file, 'r') as csvfile:
        data = []
        csv_reader = csv.reader(csvfile)
        data.append(next(csv_reader))
        for row in csv_reader:
            data.append([float(value) for value in row])
    return data


def main():
    data = make_list('Advertising.csv')

    predictor = least_squares.LS(data)
    predicted_num = predictor.predict()
    print(f'Predicted result: {predicted_num}')


if __name__ == '__main__':
    main()
