using Accord;
using ActTracker.Models;
using Docker.DotNet;
using Docker.DotNet.Models;
using Microsoft.ML;
using MoreLinq;
using PLplot;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ActTracker
{
    static class Extensions
    {
        public static List<List<T>> SplitList<T>(this List<T> me, int size = 5)
        {
            var list = new List<List<T>>();
            for (int i = 0; i < me.Count; i += size)
                list.Add(me.GetRange(i, Math.Min(size, me.Count - i)));
            return list;
        }
    }
    public static class home
    {
        static readonly string _dataPath = Path.Combine("/D/", "linacc.csv");
        static List<CustomDouble> foundRoots = new List<CustomDouble>();
        public static async Task Main(string[] args)
        {
            //init docker configurations
            ContainerListResponse pythonContainer = new ContainerListResponse(); ;
            DockerClient client = null;
            try
            {
                string socket = "unix:///var/run/docker.sock"; //linux socket
                client = new DockerClientConfiguration(
                     new Uri(socket))
                     .CreateClient();
                IList<ContainerListResponse> containers = await client.Containers.ListContainersAsync(
                    new ContainersListParameters()
                    {
                        Limit = 10,
                    });
                pythonContainer = containers.ToList().Find(x => x.Names[0].Contains("raw2count"));
            }
            catch (Exception ex)
            {
                Console.WriteLine("Machine is not running inside a docker container");
            }

            //Load in data from sensor
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<SensorData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            var accelerations = mlContext.Data.CreateEnumerable<SensorData>(dataView, reuseRowObject: false).ToList();
            var time = accelerations.Select(x => x.Time).ToList();
            //Apply filter
            List<double> difference = new List<double>();
            for (int k = 1; k < accelerations.Count; k++)
                difference.Add(accelerations[k].Time - accelerations[k - 1].Time);
            var average = difference.Sum(x => x) / difference.Count();
            var filteredAcceleration = Butterworth(accelerations.Select(x => x.Acceleration_x).ToArray(), average, 0.5);
            var sensordata = new List<Sensor>();
            for (int k = 0; k < filteredAcceleration.Count(); k++)
            {
                sensordata.Add(new Sensor { Time = Convert.ToDouble(time[k]), Data = Convert.ToDouble(filteredAcceleration[k]) }); ;
            }
            //find roots
            bisection(1, filteredAcceleration.Count() - 1, filteredAcceleration.ToList());

            //export for python script to go
            exportdata(accelerations.Select(x => x.Acceleration_x).ToList(), "Versnelling_Unfiltered", "m/s^2");

            //start container
            Console.WriteLine("starting python script...");
            if (client != null)
            {
                await client.Containers.StartContainerAsync(pythonContainer.ID, new ContainerStartParameters
                {
                }, CancellationToken.None);
            }
        }
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
        private static List<Point> createPoint(List<Sensor> l)
        {
            List<Point> p = new List<Point>();
            for (int k = 0; k < l.Count; k++)
            {
                p.Add(new Point { X = (float)l[k].Time, Y = (float)(l[k].Data) });
            }
            return p;
        }
        private static List<Sensor> Integrate(List<Point> p)
        {
            List<Sensor> FunctionValues = new List<Sensor>();
            //v = v0 + a(t)dt
            //s = s0 + v(t)dt
            for (int i = 0; i < p.Count - 1; i++)
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
            sb.AppendLine($"x");
            foreach (var item in states)
            {
                sb.AppendLine(item.ToString() + ",");
            }
            Directory.CreateDirectory(System.IO.Path.Combine("/D/", "acc"));
            Directory.CreateDirectory(System.IO.Path.Combine("/D/", "acc_export"));
            System.IO.File.WriteAllText(
                System.IO.Path.Combine("/D/", "acc", $"linacc.csv"),
                sb.ToString());
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
        static void bisection(double a,
                       double b, List<Double> data)
        {
            List<ThreePoints> three_points = new List<ThreePoints>();
            //{1,2,.... n}
            //take 3 points 
            //{1,2,3} , {2,3,4} , {3,4,5}, .... ,{n-2, n-1, n}
            //if -2 < average < 2 --> somewhat a root!
            for (int j = 2; j < data.Count; j++)
            {
                ThreePoints x = new ThreePoints();
                List<CustomDouble> z = new List<CustomDouble>();
                z.Add(new CustomDouble { FunctionValue = data[j - 2], countInArray = j - 2 });
                z.Add(new CustomDouble { FunctionValue = data[j - 1], countInArray = j - 1 });
                z.Add(new CustomDouble { FunctionValue = data[j], countInArray = j });
                var average = (z.Select(p => p.FunctionValue).Sum()) / 3;
                x.average = average;
                x.three_points = z;
                three_points.Add(x);
            }
            foreach (var k in three_points)
            {
                if (k.average < .1 && k.average > -.1) // -2 < average < 2
                {
                    //find which one in function_values is closest to zero..
                    //Min{Math.Abs(x), x = function_value}
                    var j = k.three_points.MinBy(p => Math.Abs(p.FunctionValue)).First();
                    var findWithin5 = foundRoots.Where(x => Math.Abs(x.countInArray - j.countInArray) < 40);
                    if (findWithin5.Count() == 0)
                        foundRoots.Add(j);
                }
            }
            foreach (var j in foundRoots)
                Console.WriteLine("Root : x = " + j.countInArray);

        }
    }
}
