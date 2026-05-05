import logging
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from neuralforecast.models import Autoformer, Informer, NBEATS, NHITS, TFT
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE, MAE
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from data_utils.generate_fiscal_periods import generate_periods
from data_utils.filter_export_csv import generate_fiscal_account_data
from data_utils.training_helper import period_to_h_value
from data_utils.attention_plots import (
    plot_attention,
    plot_attention_with_dates,
    plot_attention_with_dates_with_coords,
)
from data_utils.feature_importance_plots import (
    plot_future_variable_importance,
    plot_past_variable_importance,
    plot_variable_importance_means,
)


def handle_outliers_iqr(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)


def floor_lower_outliers_to_mean(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    mean_value = series.median()
    series = series.copy()
    series[series < lower_bound] = mean_value
    return series


def data(predict_year, Scenario, future_forecast, base_folder, data_cluster):

    train_start_year = "FY13"
    train_start_month = "DECEMBER"
    feature_list = ["ORGANIC NET REVENUES", "NET SALES UNITS", "GROSS SALES (4000)"]

    file_name_split = os.path.basename(__file__).replace('.py', '').split('_')
    channel = file_name_split[5]
    y_value = file_name_split[0]

    result = generate_periods(predict_year, Scenario, bool(future_forecast))

    data_df_train = generate_fiscal_account_data(
        data_cluster[file_name_split[4]],
        file_name_split[6],
        channel,
        'ACTUAL',
        train_start_year,
        train_start_month,
        result['train_end_year'],
        result['train_end_month'],
        feature_list
    )

    if bool(future_forecast):
        data_df_test = generate_fiscal_account_data(
            data_cluster[file_name_split[4]],
            file_name_split[6],
            channel,
            result['future_forecast'],
            result['test_start_year'],
            result['test_start_month'],
            result['test_end_year'],
            result['test_end_month'],
            feature_list
        )
        data_df = pd.concat([data_df_train, data_df_test], ignore_index=True)
    else:
        data_df = data_df_train

    data_df = data_df.rename(columns={y_value: 'y'})
    data_df['ds'] = pd.to_datetime(data_df['ds'], dayfirst=True)
    data_df['year'] = data_df['ds'].dt.year
    data_df['month'] = data_df['ds'].dt.month

    covid_mask = (data_df['year'] == 2020) & (data_df['month'] >= 3) & (data_df['month'] <= 11)

    monthly_mean_y = (
        data_df[data_df['year'] != 2020]
        .groupby('month')['y']
        .mean()
    )

    data_df.loc[covid_mask, 'y'] = data_df.loc[covid_mask, 'month'].map(monthly_mean_y)
    data_df.drop(columns=['year', 'month'], inplace=True)

    training_file_name = (
        base_folder + '/training_data/' + predict_year + '/' +
        os.path.basename(__file__).replace('.py', '') + '_' + Scenario + '.csv'
    )

    data_df.to_csv(training_file_name, index=False)
    return data_df, y_value


def main(data, predict_year, Scenario, future_forecast, base_folder):
    FORECAST_START_DATE = "2025-06-01"
    plots_dir = os.path.join(base_folder, "plots", predict_year, Scenario)
    os.makedirs(plots_dir, exist_ok=True)

    remove_number, h_value = period_to_h_value(Scenario)
    header_rows = data.columns.tolist()
    filtered_headers = [col for col in header_rows if col not in ['ds', 'unique_id', 'y']]
    data['ds'] = pd.to_datetime(data['ds'], dayfirst=True)

    train_df = data.iloc[:remove_number]
    test_df = data.iloc[remove_number:]
    future_df = test_df.drop(columns=['y'], axis=1)

    modelgc = TFT(
        h=3,
        input_size=12,
        hidden_size=48,
        dropout=0.1,
        n_head=8,
        learning_rate=0.001,
        batch_size=32,
        max_steps=100,
        val_check_steps=50,
        early_stop_patience_steps=50,
        scaler_type='standard',
        loss=MAE(),
        futr_exog_list=filtered_headers
    )

    nf = NeuralForecast(models=[modelgc], freq='ME')
    nf.fit(df=train_df, val_size=24)
    preds = nf.predict(futr_df=future_df)
    print(preds)

    feature_importances = nf.models[0].feature_importances()

    # --- Feature importance plots ---
    plot_future_variable_importance(feature_importances, plots_dir, FORECAST_START_DATE)
    plot_past_variable_importance(feature_importances, plots_dir, FORECAST_START_DATE)
    plot_variable_importance_means(feature_importances, plots_dir)

    # --- Attention plots ---
    plot_attention(nf.models[0], plots_dir, plot="time")
    plot_attention(nf.models[0], plots_dir, plot="all")
    plot_attention(nf.models[0], plots_dir, plot="heatmap")
    plot_attention(nf.models[0], plots_dir, plot=1)

    plot_attention_with_dates(
        tft_model=nf.models[0],
        plots_dir=plots_dir,
        forecast_start_date=FORECAST_START_DATE,
    )

    plot_attention_with_dates_with_coords(
        tft_model=nf.models[0],
        plots_dir=plots_dir,
        forecast_start_date=FORECAST_START_DATE,
    )

    output_file_name = (
        base_folder + '/output_data/' + predict_year + '/' +
        os.path.basename(__file__).replace('.py', '') + '_' + Scenario + '.csv'
    )
    preds.to_csv(output_file_name, index=False)
    return preds


if __name__ == "__main__":

    clusters = ['GREATER CHINA']
    cluster_data = ['finance/Data/GREATER CHINA_cluster_data_plan.csv']
    plots_dir = "finance/Data/output_data/FY25"
    data_cluster = {}
    for i in range(len(clusters)):
        data_cluster.update({clusters[i]: pd.read_csv(cluster_data[i])})

    Year = "FY25"
    Scenario = "P7"
    future_forecast = True
    base_folder = "finance"

    data_df, y_value = data(Year, Scenario, future_forecast, base_folder, data_cluster)
    main(data_df, Year, Scenario, future_forecast, base_folder)

    if os.path.exists('lightning_logs'):
        shutil.rmtree('lightning_logs')
