import pandas as pd
import numpy as np

def stratified_split(df: pd.DataFrame,
                     y_col: str,
                     split: float = 0.8,
                     seed: int = 1337):
    """Split data into stratified (same proportion of classes) train/test split

    Adapted from previous mini-project.

    :param df: dataframe containing data
    :param y_col: dataframe column containing class labels
    :param split: fraction of data to use for training
    :param seed: random seed to ensure replicability
    :return:
    """
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df = df.sample(frac=1, random_state=seed)  # shuffle
    for y_class in df[y_col].unique():
        df_y_class = df.loc[df[y_col] == y_class]
        i_split = int(len(df_y_class) * split)
        df_train = df_train.append(df_y_class[:i_split])
        df_test = df_test.append(df_y_class[i_split:])
    return df_train, df_test


def stratified_k_folds(df: pd.DataFrame,
                       y_col: str,
                       k: int = 5,
                       seed: int = 1337):
    """Split data into stratified (same proportion of classes) cross-validation splits.

    Adapted from previous mini-project.

    :param df: dataframe containing data
    :param y_col: dataframe column containing class labels
    :param k: number of folds
    :param seed: random seed to ensure replicability
    :return:
    """
    df = df.sample(frac=1, random_state=seed)  # shuffle
    df_train = [pd.DataFrame() for _ in range(k)]
    df_test = [pd.DataFrame() for _ in range(k)]
    for y_class in df[y_col].unique():
        df_y_class = df.loc[df[y_col] == y_class]
        df_y_class.reset_index(inplace=True)
        k_eval_size = len(df_y_class) // k
        for k_i in range(k):
            i_start = k_eval_size * k_i
            i_end = i_start + k_eval_size
            i_train = list(range(0, i_start)) + list(
                range(i_end, len(df_y_class)))
            i_eval = list(range(i_start, i_end))
            df_train[k_i] = df_train[k_i].append(df_y_class.iloc[i_train])
            df_test[k_i] = df_test[k_i].append(df_y_class.iloc[i_eval])
    return df_train, df_test

def evaluate_acc(target_y, true_y):
    '''
    Compute the accuracy of the model
    :param target_y: predictions from the model
    :param true_y: true categories
    :return: the accuracy

    adapted from previous mini project
    '''
    correct_labels = 0
    if len(target_y) != len(true_y):  # to prevent indexing exceptions
        print("can't compare those sets, not the same size")
        return -1  # return error code
    for i in range(len(target_y)):
        if target_y[i] == true_y[i]:
            correct_labels += 1  # we count how many labels the model got right
    return correct_labels/len(target_y)  # we return the ratio over correct over total


def evaluate(predicted_y, true_y, label_stoi):
    y = np.zeros((len(true_y), len(label_stoi)))
    for i, l in enumerate(true_y):
        y[i, label_stoi[l]] = 1

    y_pred = np.zeros((len(predicted_y), len(label_stoi)))
    for i, l in enumerate(predicted_y):
        y_pred[i, label_stoi[l]] = 1

    total = len(y)
    tp = ((y == 1) & (y_pred == 1)).sum(0)
    fp = ((y == 0) & (y_pred == 1)).sum(0)
    tn = ((y == 0) & (y_pred == 0)).sum(0)
    fn = ((y == 1) & (y_pred == 0)).sum(0)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return {
        label: {
            "a": accuracy[i],
            "r": recall[i],
            "p": precision[i],
        } for label,i in label_stoi.items()
    }


