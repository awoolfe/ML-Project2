import pandas as pd

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

