import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

model_class = {
    "linear regression": LinearRegression,
    "random forest": RandomForestRegressor
}
save_name = {
    "linear regression": "lr.csv",
    "random forest": "rf.csv"
}

def eval_model(model_name):
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible
    """

    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3  # Modify this value to change the number of repetitions
    train_frac = 0.7  # Modify this value to change the training data fraction (e.g., 0.7 for 70%)
    random_seed = 1 # The random seed will be altered for each repeat
    result_df = pd.DataFrame(columns=["System", "Dataset", "MAPE", "MAE", "RMSE"])

    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system) # Modify this to specify the location of the datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')] # List all CSV files in the directory

        for csv_file in csv_files:
            # print('\n> System: {}, Dataset: {}, Training data fraction: {}, Number of repets: {}'.format(current_system, csv_file, train_frac, num_repeats))

            data = pd.read_csv(os.path.join(datasets_location, csv_file)) # Load data from CSV file

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []} # Initialize a dict to store results for repeated evaluations

            for current_repeat in range(num_repeats): # Repeat the process n times
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat) # Change the random seed based on the current repeat
                test_data = data.drop(train_data.index)

                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                model = model_class[model_name]()

                model.fit(training_X, training_Y) # Train the model with the training data

                predictions = model.predict(testing_X) # Predict the testing data

                # Calculate evaluation metrics for the current repeat
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Store the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Calculate the average of the metrics for all repeats
            # print('Average MAPE: {:.2f}'.format(np.mean(metrics['MAPE'])))
            # print("Average MAE: {:.2f}".format(np.mean(metrics['MAE'])))
            # print("Average RMSE: {:.2f}".format(np.mean(metrics['RMSE'])))
            result_df = result_df._append({
                "System": current_system, 
                "Dataset": csv_file, 
                "MAPE": np.mean(metrics["MAPE"]), 
                "MAE": np.mean(metrics["MAE"]), 
                "RMSE": np.mean(metrics["RMSE"])
            }, ignore_index=True)
    
    # result_df.to_csv(save_name[model_name])
    return result_df


def stats_analysis(lr_result, rf_result):
    # visualization
    for metric in ["MAPE", "MAE", "RMSE"]:
        plt.scatter(lr_result[metric], rf_result[metric], c=(rf_result[metric]>lr_result[metric]), cmap="coolwarm")
        min_x = min(lr_result[metric].min(), rf_result[metric].min())
        max_x = max(lr_result[metric].max(), rf_result[metric].max())
        plt.loglog([min_x, max_x], [min_x, max_x], color="black", linestyle="dashed")
        plt.xlabel("Linear regression")
        plt.ylabel("Random forest")
        plt.xscale("log")
        plt.yscale("log")
        plt.title(metric)
        plt.savefig("{}.png".format(metric))
        plt.clf()

    # pairwise t-test analysis
    t_mape = ttest_rel(lr_result["MAPE"], rf_result["MAPE"])
    t_mae = ttest_rel(lr_result["MAE"], rf_result["MAE"])
    t_rmse = ttest_rel(lr_result["RMSE"], rf_result["RMSE"])
    print(t_mape.pvalue, t_mae.pvalue, t_rmse.pvalue)

if __name__ == "__main__":
    lr_result = eval_model("linear regression")
    rf_result = eval_model("random forest")
    stats_analysis(lr_result, rf_result)
