using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace HouseRegression
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }
    }
}
