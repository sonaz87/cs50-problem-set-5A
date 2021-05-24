import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number   9
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer   11
        - Browser, an integer
        - Region, an integer   13
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    evidence = []
    labels = []
    months = {'Jan': 1, 'Feb' : 2, 'Mar': 3, 'May' : 5, 'June' : 6, 'Jul' : 7, 'Aug' : 8, 'Sep' : 9, 'Oct' : 10, 'Nov' : 11, 'Dec' : 12}
    
    
    
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            evidence.append([int(line[0]), float(line[1]), int(line[2]), float(line[3]), int(line[4]), float(line[5]), 
                             float(line[6]), float(line[7]), float(line[8]), 
                             float(line[9]), months[line[10]], int(line[11]), int(line[12]), int(line[13]), int(line[14]), 
                             0 if line[15] == 'New_Visitor' else 1, 0 if line[16] == 'FALSE' else 1 ])
            labels.append(0 if line[17] == 'FALSE' else 1)
            
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 1)
    X_training = [row for row in evidence]
    y_training = [row for row in labels]
    result = model.fit(X_training, y_training)
    
    return result
    


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """


    
    # Compute how well we performed
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_pos += 1
            if  actual == predicted:
                correct_pos += 1
        elif actual == 0:
            total_neg += 1
            if actual == predicted:
                correct_neg += 1

    return correct_pos/total_pos, correct_neg/total_neg

if __name__ == "__main__":
    main()

