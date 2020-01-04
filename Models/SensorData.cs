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
    public class WheelData
    {
        [LoadColumn(0)]
        public double Time;
        [LoadColumn(1)]
        public double GyorX;
        [LoadColumn(2)]
        public double GyroY;
        [LoadColumn(3)]
        public double GyorZ;
        [LoadColumn(4)]
        public double Accx;
        [LoadColumn(5)]
        public double Accy;
        [LoadColumn(6)]
        public double Accz;
        [LoadColumn(4)]
        public double Bx;
        [LoadColumn(5)]
        public double By;
        [LoadColumn(6)]
        public double Bz;
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
}
