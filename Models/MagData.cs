using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Globalization;

namespace ActTracker.Models
{
    public class MagData
    {
        public float[] Gyroscope { get; set; }
        public float[] Accelerometer { get; set; }
        public float[] Magnetometer { get; set; }

        public MagData()
            : this(new float[3] { 0, 0, 0 }, new float[3] { 0, 0, 0 }, new float[3] { 0, 0, 0 })
        {
        }
        public MagData(float[] gyroscope, float[] accelerometer, float[] magnetometer)
        {
            Gyroscope = gyroscope;
            Accelerometer = accelerometer;
            Magnetometer = magnetometer;
        }
    }
}
