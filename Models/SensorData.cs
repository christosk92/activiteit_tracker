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
    public class linnaccpy
    {
        [LoadColumn(0)]
        public double axis1;
    }
    public class Sensor
    {
        public double Time;
        public double Data;
    }
    public class EulerData
    {
        [LoadColumn(0)]
        public double Z;
        [LoadColumn(1)]
        public double Y;
        [LoadColumn(2)]
        public double X;
    }
}
