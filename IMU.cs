using Iot.Device.CpuTemperature;
using System;
using Microsoft.Azure.Devices.Client;
using Newtonsoft.Json;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML;
using System.Linq;
using PLplot;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace ActTracker
{
    class IMU
    {
        public class AccelerationRecord
        {
            [LoadColumn(0)] public string time;
            [LoadColumn(1)] public float accx;
        }
        public class AccelerationPrediction
        {
            //vector to hold alert,score,p-value values
            [VectorType(3)] public double[] Prediction { get; set; }
        }
        public class OutputSpike
        {
            public string time { get; set; }
            public float acc { get; set; }
            public override string ToString()
            {
                return this.time + "," + this.acc.ToString();
            }
        }
        private static string dataPath = Path.Combine(@"D", "y.csv");
        
        static List<OutputSpike> DetectSpike(MLContext mlContext, int docSize, IDataView accelerationData)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(AccelerationPrediction.Prediction), inputColumnName: nameof(AccelerationRecord.accx), confidence: 80, pvalueHistoryLength: docSize / 4);
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
            IDataView transformedData = iidSpikeTransform.Transform(accelerationData);
            var predictions = mlContext.Data.CreateEnumerable<AccelerationPrediction>(transformedData, reuseRowObject: false).ToList() ;
            var date = accelerationData.GetColumn<string>("time").ToArray();
            var value = accelerationData.GetColumn<float>("accx").ToArray();
            List<OutputSpike> pSpikes = new List<OutputSpike>();
            Console.WriteLine("Alert\tTime\tValue");
            for (int i = 0; i < predictions.Count; i++)
            {
                if (predictions[i].Prediction[0] == 1)
                {
                    var results = $"{predictions[i].Prediction[0]}\t{date[i]:f2}\t{value[i]:F2}";
                    results += " <-- Spike detected";
                    Console.WriteLine(results);
                    var output = new OutputSpike
                    {
                        acc = value[i],
                        time = date[i]
                    };
                    pSpikes.Add(output);
                }
            }
            //save as csv
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            sb.AppendLine("time,acc");
            foreach (var item in pSpikes)
            {
                sb.AppendLine(item.ToString());
            }
            string path = System.IO.Path.Combine("D", "pspikes.csv");
            if (System.IO.File.Exists(path))
                System.IO.File.Delete(path);
            
            System.IO.File.WriteAllText(
                System.IO.Path.Combine(path),
                sb.ToString());
            return pSpikes;
        }

        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Create empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<AccelerationRecord> enumerableData = new List<AccelerationRecord>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }
        public struct Measurement
        {
            private double variance;
            public double Value { get; set; }
            public string Time { get; set; }
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
        private static Random rnd = new Random();

        public static double[] SineWave(int iteration)
        {
            return new[] { Math.Sin(iteration * 3.14 * 5 / 180) + (double)rnd.Next(50) / 100 };
        }
        public static ObservableCollection<Measurement> Measurements { get; set; }
        public static ObservableCollection<Measurement> Estimates { get; set; }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<AccelerationRecord>(path: dataPath, hasHeader: true, separatorChar: ',');

            var Yacceleration = mlContext.Data.CreateEnumerable<AccelerationRecord>(dataView, reuseRowObject: false).ToList();

            var predictions = DetectSpike(mlContext, Yacceleration.Count, dataView);

            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("y.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                 0, 5,                          // x-axis range
                 -10, 15,                         // y-axis range
                 AxesScale.Independent,          // scale x and y independently
                 AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "t: s",                         // x-axis label
                "Y: m/s^2",                        // y-axis label
                "Y-Acceleration");     // plot title
           /* pl.line(
                (from x in Yacceleration select (double)Convert.ToDouble(x.time)).ToArray(),
                (from p in Yacceleration select (double)p.accx).ToArray()
            );

            var pepega = (from i in Enumerable.Range(0, predictions.Count())
                          select (Time: predictions[i].time, Acceleration: predictions[i].acc));

            // plot the spikes
            pl.col0(2);     // blue color
            pl.string2(
                (from s in pepega select (double)Convert.ToDouble(s.Time)).ToArray(),
                (from s in pepega select (double)s.Acceleration).ToArray(),
                "!");
                */
            Measurements = new ObservableCollection<Measurement>();
            Estimates = new ObservableCollection<Measurement>();

            var filter = new UKF();
            var N = Yacceleration.Count;


            for (int k = 0; k < N; k++)
            {
                double[] z = { Yacceleration[k].accx };
                filter.Update(z);
                var state = filter.getState();
                var covariance = filter.getCovariance();

                Measurements.Add(new Measurement() { Value = z[0], Time = Yacceleration[k].time });
                Estimates.Add(new Measurement() { Value = state[0], Time = Yacceleration[k].time, Variance = covariance[0, 0] });
            }

            pl.line(
               (from t in Estimates select (double)Convert.ToDouble(t.Time)).ToArray(),
               (from y in Estimates select (double)y.Value).ToArray()
           ); 
            pl.col0(2);
            pl.line(
              (from t in Measurements select (double)Convert.ToDouble(t.Time)).ToArray(),
              (from y in Measurements select (double)y.Value).ToArray()
          );
            pl.eop();


        }

    }
}
