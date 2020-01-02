using Accord;
using Accord.Math;
using Microsoft.ML;
using Microsoft.ML.Data;
using MoreLinq;
using PLplot;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ActTracker
{
    public class SensorData
    {
        [LoadColumn(0)]
        public double Time;

        [LoadColumn(3)]
        public double Acceleration_x;
    }
    public class SensorDataPrediction
    {
        //vector to hold alert,score,p-value values
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
    public class Sensor
    {
        public double Time;
        public double Data;
    }

    class IMU
    {
        public static double[] Butterworth(double[] indata, double deltaTimeinsec, double CutOff)
        {
            if (indata == null) return null;
            if (CutOff == 0) return indata;

            double Samplingrate = 1 / deltaTimeinsec;
            long dF2 = indata.Length - 1;        // The data range is set with dF2
            double[] Dat2 = new double[dF2 + 4]; // Array with 4 extra points front and back
            double[] data = indata; // Ptr., changes passed data

            for (long r = 0; r < dF2; r++)
            {
                Dat2[2 + r] = indata[r];
            }
            Dat2[1] = Dat2[0] = indata[0];
            Dat2[dF2 + 3] = Dat2[dF2 + 2] = indata[dF2];

            const double pi = Math.PI;
            double wc = Math.Tan(CutOff * pi / Samplingrate);
            double k1 = Math.Sqrt(2) / 2 * wc;
            double k2 = wc * wc;
            double a = k2 / (1 + k1 + k2);
            double b = 2 * a;
            double c = a;
            double k3 = b / k2;
            double d = -2 * a + k3;
            double e = 1 - (2 * a) - k3;

            // RECURSIVE TRIGGERS - ENABLE filter is performed (first, last points constant)
            double[] DatYt = new double[dF2 + 4];
            DatYt[1] = DatYt[0] = indata[0];
            for (long s = 2; s < dF2 + 2; s++)
            {
                DatYt[s] = a * Dat2[s] + b * Dat2[s - 1] + c * Dat2[s - 2]
                           + d * DatYt[s - 1] + e * DatYt[s - 2];
            }
            DatYt[dF2 + 3] = DatYt[dF2 + 2] = DatYt[dF2 + 1];

            // FORWARD filter
            double[] DatZt = new double[dF2 + 2];
            DatZt[dF2] = DatYt[dF2 + 2];
            DatZt[dF2 + 1] = DatYt[dF2 + 3];
            for (long t = -dF2 + 1; t <= 0; t++)
            {
                DatZt[-t] = a * DatYt[-t + 2] + b * DatYt[-t + 3] + c * DatYt[-t + 4]
                            + d * DatZt[-t + 1] + e * DatZt[-t + 2];
            }

            // Calculated points copied for return
            for (long p = 0; p < dF2; p++)
            {
                data[p] = DatZt[p];
            }

            return data;
        }

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "raw.csv");

        public static long? _docsize = 0;
        private static System.Timers.Timer aTimer;
        private static List<double> corrected = new List<double>();
        private static List<CustomBool> maxOrMins = new List<CustomBool>();
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<SensorData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            var accelerations = mlContext.Data.CreateEnumerable<SensorData>(dataView, reuseRowObject: false).ToList();
            var time = accelerations.Select(x => x.Time).ToList();
            List<Sensor> sensordata = new List<Sensor>();
            exportdata(accelerations.Select(x => x.Acceleration_x).ToList(), "Versnelling_Unfiltered", "m/s^2");
            List<double> difference = new List<double>();
            for (int k = 1; k < accelerations.Count; k++)
                difference.Add(accelerations[k].Time - accelerations[k - 1].Time);
            var average = difference.Sum(x => x) / difference.Count();
            var filteredAcceleration = Butterworth(accelerations.Select(x => x.Acceleration_x).ToArray(), average, 2);
            exportdata(filteredAcceleration.ToList(), "Versnelling", "m/s^2");
            sensordata = new List<Sensor>();
            for (int k = 0; k < filteredAcceleration.Count(); k++)
            {
                sensordata.Add(new Sensor { Time = Convert.ToDouble(time[k]), Data = Convert.ToDouble(filteredAcceleration[k]) }); ;
            }
            var jerk = Derivative(sensordata);
            originalDataDont = jerk.ToList();
            bisection(1, jerk.Count() - 1, jerk);

            for (int i = 0; i < foundRoots.Count - 1; i += 15)
            {
                var z = foundRoots.Where(x => Math.Abs(x.countInArray - i) < 15); //find all poins that are a distance 20 away from the current root.
                if (z.Count() > 0)
                {
                    var minValues = z.ToArray().MinBy(x => Math.Abs((double)x.FunctionValue));
                    var jk = minValues.Aggregate((curMin, x) => (curMin == null || (x.FunctionValue) <
                        curMin.FunctionValue ? x : curMin));
                    var existsInRealArray = arrayofroots.FindIndex(x => x == jk.countInArray);
                    if (jk.countInArray != previousRoot)
                    {
                        int h = 3;
                        if (((originalDataDont[jk.countInArray + h]) > 0 && (originalDataDont[jk.countInArray - h]) < 0) || ((originalDataDont[jk.countInArray + h]) < 0 && (originalDataDont[jk.countInArray - h]) > 0))
                        {
                            arrayofroots.Add(jk.countInArray);
                            bool max = false;
                            if ((originalDataDont[jk.countInArray + h]) > 0 && (originalDataDont[jk.countInArray - h]) < 0)
                                max = false;
                            else
                                max = true;
                            maxOrMins.Add(new CustomBool { index = jk.countInArray, Max = max});
                            previousRoot = jk.countInArray;
                        }
                    }
                }
            }
            List<CustomDouble2> Force = new List<CustomDouble2>();
            foreach (var k in maxOrMins.Distinct())
            {
                //find value in the array...
                string type = "";
                if (k.Max)
                    type = "Maximum";
                else
                    type = "Minimum";
                Console.WriteLine($"Sample {k.index} is a {type}");
            } 
            //F = ma... 
            double m = 80;
            for (int j = 0; j < maxOrMins.Count - 1; j++)
            {
                if (maxOrMins[j].Max)
                {
                    Force.Add(new CustomDouble2 { countEnd = (int)maxOrMins[j + 1].index, countStart = (int)maxOrMins[j].index, Value = m * (filteredAcceleration.ToList()[(int)maxOrMins[j + 1].index] - filteredAcceleration.ToList()[(int)maxOrMins[j].index]) });
                }
            }

            jerk = Derivative(sensordata);
            exportdata(jerk.ToList(), "Jerk", "m/s^3", arrayofroots.Distinct().ToList());
            var velocity = Integrate(createPoint(sensordata));
            var filteredVelocity = Butterworth(velocity.Select(x => x.Data).ToArray(), average, 0.1);
            exportdata(filteredVelocity.ToList(), "Velocity", "m/s");

            exportdata(filteredAcceleration.ToList(), "Versnelling", "m/s^2", arrayofroots.Distinct().ToList());

            sensordata = new List<Sensor>();
            for (int k = 0; k < filteredVelocity.Count(); k++)
            {
                sensordata.Add(new Sensor { Time = Convert.ToDouble(time[k]), Data = Convert.ToDouble(filteredVelocity[k]) }); ;
            }
            var distance = Integrate(createPoint(sensordata));

            List<double> WorkDone = new List<double>();
            for (int k = 0; k < Force.Count; k++)
            {
                //b-a on a time interval...
                //Polling frequency = 100 Hz --> 1 sample = 0.01s
                var PositionB = distance[Force[k].countEnd].Data;
                var PositionA = distance[Force[k].countStart].Data;
                WorkDone.Add(Math.Abs(Force[k].Value) * Math.Abs((PositionB - PositionA)));
            }
            //1 J = 0.000239 KCal
            //x J = y kCal
            //y = x * 0.000239 / 1
            Console.WriteLine("Work done: " + WorkDone.Sum(x => x) * 0.000239 + " KCal");
            var filteredDistance = Butterworth(distance.Select(x => x.Data).ToArray(), average, 0.1);
            exportdata(filteredDistance.ToList(), "Position", "m");
        }
        public class CustomBool
        {
            public double index { get; set; }
            public bool Max { get; set; }
        }
        static float EPSILON = (float)0.0001;

        static List<int> arrayofroots = new List<int>();

        static List<CustomDouble> foundRoots = new List<CustomDouble>();

        static int previousRoot = 0;
        static List<Double> originalDataDont = new List<double>();
        static void bisection(double a,
                           double b, List<Double> data)
        {
            var actualOriginal = a; // dont change
            var actualOriginalB = b; // dont change
            var originalA = a;
            var originalB = b;
            var j = data[(int)Math.Round(a, 0)];
            var k = data[(int)Math.Round(b, 0)];
            bool invalidRoot = true;
            while (invalidRoot)
            {
                if (originalB > originalA)
                {
                    if (j * k >= 0)
                    {
                        j = data[(int)Math.Round(originalA, 0)];
                        k = data[(int)Math.Round(originalB - 1, 0)];
                        originalB -= 1;
                    }
                    else
                        invalidRoot = false;
                }
                else
                    break;
            }
            if (originalB > originalA)
            {
                if (j * k >= 0)
                {
                    Console.WriteLine("wrong a and b");
                    return;
                }
                double c = a;
                while ((b - a) >= EPSILON)
                {
                    // Find middle point 
                    c = (a + b) / 2;
                    var nearest = data.ToArray().MinBy(x => Math.Abs((double)x - c)).ToList();
                    if (Math.Round(nearest.First(), 3) == 0.000)
                        break;
                    else if (nearest.First() * data[(int)Math.Round(a, 3)] < 0)
                        b = c;
                    else
                        a = c;
                }
                // prints value of c  
                // upto 4 decimal places
                double nearestDouble = 0.0;
                foundRoots.Add(new CustomDouble { countInArray = (int)Math.Round(c, 0), FunctionValue = originalDataDont[(int)Math.Round(c, 0)] });
                int xroot = (int)Math.Round(c, 3);
                var newData = data;
                newData.Remove(newData[xroot]);
                bisection(actualOriginal, newData.Count - 1, newData);
            }
        }
        class CustomDouble
        {
            public int countInArray { get; set; }
            public double FunctionValue { get; set; }
        }
        class CustomDouble2
        {
            public int countStart { get; set; }
            public int countEnd { get; set; }
            public double Value { get; set; }
        }
        static List<double> Derivative(List<Sensor> args)
        {
            List<double> jp = new List<double>();
            for (int i = 0; i < args.Count; i++)
            {
                if (i != 0)
                {
                    var dt = args[i].Time - args[i - 1].Time;
                    var result = (args[i].Data - args[i - 1].Data) / dt;
                    jp.Add(result);
                }
                else
                {
                    jp.Add(0);

                }
            }
            return jp;
        }

        private static List<Point> createPoint(List<Sensor> l)
        {
            List<Point> p = new List<Point>();
            for (int k = 0; k < l.Count; k++)
            {
                p.Add(new Point { X = (float)l[k].Time, Y = (float)(l[k].Data) });
            }
            return p;
        }
        private static List<Sensor> function(int count)
        {
            var lk = new List<Sensor>();
            for (int j = 0; j < count + 1; j++)
            {
                var result = j * j;
                lk.Add(new Sensor { Data = result, Time = j });
            }
            return lk;
        }
        private static List<Sensor> Integrate(List<Point> p)
        {
            List<Sensor> FunctionValues = new List<Sensor>();
            //v = v0 + a(t)dt
            //s = s0 + v(t)dt
            for (int i = 0; i < p.Count; i++)
            {
                if (i != 0)
                {
                    FunctionValues.Add(new Sensor { Data = (p[i].Y + p[i - 1].Y) / 2 * (p[i].X - p[i - 1].X) + FunctionValues[i - 1].Data, Time = p[i].X });
                }
                else
                {
                    FunctionValues.Add(new Sensor { Data = 0, Time = 0 });
                }
            }
            return FunctionValues;
        }

        private static void exportdata(List<double> states, string name, string unit, List<int> second = null)
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            sb.AppendLine($"{name}");
            foreach (var item in states)
            {
                sb.AppendLine(item.ToString() + ",");
            }

            System.IO.File.WriteAllText(
                System.IO.Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory, $"{name}.csv"),
                sb.ToString());
            var test = states.Min(x => x);
            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam($"{name}.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, states.Count,                          // x-axis range
                 states.Min(x => x), states.Max(x => x),                         // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Sample",                         // x-axis label
                unit,                        // y-axis label
                name);     // plot title
            pl.line(
                (from x in Enumerable.Range(0, states.Count()) select (double)x).ToArray(),
                (from p in states select (double)p).ToArray()
            );
            if (second != null)
            {
                pl.col0(4);
                var Roots = GetListFromIndices(second, states);
                pl.poin(
                 (from x in second select (double)x).ToArray(),
                 (from p in Roots select (double)p).ToArray(),
                 '!'
             );
            }
            string csv = String.Join(",", states.Select(x => x.ToString()).ToArray());
            pl.eop();


        }

        static List<double> GetListFromIndices(List<int> indices, List<double> source)
        {
            List<double> toREturn = new List<double>();
            foreach (var k in indices)
            {
                toREturn.Add(source[k]);
            }
            return toREturn;
        }

        public static double[] functionValues(int count, double a, double b)
        {
            List<double> arr = new List<double>();
            for (int j = 0; j < count + 1; j++)
                arr.Add(a * j + b);
            return arr.ToArray();
        }
        public struct Measurement
        {
            private double variance;
            public double Value { get; set; }
            public TimeSpan Time { get; set; }
            public double Variance
            {
                get
                {
                    return variance;
                }
                set
                {
                    variance = value;
                    UpperDeviation = Value + Math.Sqrt(variance);
                    LowerDeviation = Value - Math.Sqrt(variance);
                }
            }
            public double UpperDeviation { get; private set; }
            public double LowerDeviation { get; private set; }
        }
    }
}
