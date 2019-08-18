using Microsoft.ML;
using System;
using System.Linq;

namespace HouseRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var preview = data.Preview();

            var features = data.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "OceanProximity")
                .ToArray();

            var dataPrepPipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features));

            var preppedData = dataPrepPipeline.Fit(data);

            var trainer = context.Regression.Trainers.LbfgsPoissonRegression();

            var pipeline = dataPrepPipeline.Append(trainer);

            var model = pipeline.Fit(data);

            // Save full model
            context.Model.Save(model, data.Schema, "./housing-model.zip");

            // Save data prep
            context.Model.Save(preppedData, data.Schema, "./housing-data-prep.zip");

            // Save trainer
            var transformedData = preppedData.Transform(data);
            var transformedTrainer = trainer.Fit(transformedData);
            context.Model.Save(transformedTrainer, transformedData.Schema, "./housing-trainer.zip");

            var predictionFunc = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

            var prediction = predictionFunc.Predict(new HousingData
            {
                Longitude = -122.25f,
                Latitude = 37.85f,
                HousingMedianAge = 55.0f,
                TotalRooms = 1627.0f,
                TotalBedrooms = 235.0f,
                Population = 322.0f,
                Households = 120.0f,
                MedianIncome = 8.3014f,
                OceanProximity = "NEAR BAY"
            });
        }
    }
}
