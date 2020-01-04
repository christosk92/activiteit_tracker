using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ActTracker.Models
{
    public class SensorData
    {
        [LoadColumn(0)]
        public double Time;

        [LoadColumn(1)]
        public double Acceleration_x;
    }
    public class Sensor
    {
        public double Time;
        public double Data;
    }
}
