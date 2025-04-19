#ifndef PREDICTIVE_MODELING_H
#define PREDICTIVE_MODELING_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Predictive Model Structure
typedef struct {
    double* historical_data;
    uint64_t data_count;
    uint64_t max_data_points;
} PredictiveModel;

// Function Prototypes
PredictiveModel* init_predictive_model(uint64_t max_data_points);
void add_data_point(PredictiveModel* model, double value);
double forecast_next_value(PredictiveModel* model);
void destroy_predictive_model(PredictiveModel* model);

#endif // PREDICTIVE_MODELING_H

#include "predictive_modeling.h"

// Initialize the Predictive Model
PredictiveModel* init_predictive_model(uint64_t max_data_points) {
    PredictiveModel* model = (PredictiveModel*)malloc(sizeof(PredictiveModel));
    if (!model) return NULL;

    model->historical_data = (double*)malloc(sizeof(double) * max_data_points);
    if (!model->historical_data) {
        free(model);
        return NULL;
    }

    model->data_count = 0;
    model->max_data_points = max_data_points;

    return model;
}

// Add a Data Point to the Model
void add_data_point(PredictiveModel* model, double value) {
    if (!model || model->data_count >= model->max_data_points) return;

    model->historical_data[model->data_count++] = value;
    printf("Added data point: %.2f (Total points: %lu)\n", value, model->data_count);
}

// Forecast the Next Value Using Simple Linear Regression
double forecast_next_value(PredictiveModel* model) {
    if (!model || model->data_count < 2) {
        printf("Insufficient data for forecasting.\n");
        return 0.0;
    }

    // Calculate linear regression coefficients
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (uint64_t i = 0; i < model->data_count; i++) {
        sum_x += i;
        sum_y += model->historical_data[i];
        sum_xy += i * model->historical_data[i];
        sum_x2 += i * i;
    }

    double n = (double)model->data_count;
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    // Predict the next value
    double next_x = (double)model->data_count;
    double forecast = slope * next_x + intercept;

    printf("Forecasted next value: %.2f\n", forecast);
    return forecast;
}

// Destroy the Predictive Model
void destroy_predictive_model(PredictiveModel* model) {
    if (model) {
        free(model->historical_data);
        free(model);
    }
}

