import numpy as np


def evaluate_model(model, train, test):
    """
    Evaluate a single fitted model (data train and test)
    """
    # walk-forward validation over each week
    history = train
    predictions = list()
    rmses = list()
    for i in range(len(test.x)):
        # predict the week
        yhat, rmse = model.make_predictions(history)
        # store the predictions
        predictions.append(yhat)
        rmses.append(rmse)
        # get real observation and add to history for predicting the next week
        history.x = np.vstack((history.x, test.x[i:i + 1:, ]))
        history.y = np.vstack((history.y, test.y[i:i + 1:, ]))
    # evaluate predictions days for each week
    # predictions = np.array(predictions)
    return predictions, rmses


def evaluate_model_n_times(model, train, test, n_repeat, verbose=0):
    """
    Fit and Evaluate a single model (data train and test) multiple times
    """
    regressor_list = list()
    y_hat_list = list()
    rmse_list = list()
    for i in range(n_repeat):
        regressor_list.append(model)
        regressor_list[i].fit_model(train.x, train.y, verbose)
        y_hat, rmse = evaluate_model(regressor_list[i], train, test)
        y_hat_list.append(y_hat)
        rmse_list.append(rmse)

    return list(zip(regressor_list, y_hat_list, rmse_list))


def evaluate_n_models_n_times(models_list, train, test, n_repeat, verbose=0):
    """
    Fit and Evaluate multiple models (data train and test), for multiple times
    """
    regressors_list = list()

    for model in models_list:
        regressors_list.append(evaluate_model_n_times(n_repeat, model, train, test, verbose))

    return regressors_list
