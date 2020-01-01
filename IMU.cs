using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using System;
using System.Collections.Generic;
using PLplot;

namespace ActTracker
{
    public class SensorData
    {
        [LoadColumn(0)]
        public string Time;

        [LoadColumn(1)]
        public double Acceleration_x;
    }
    public class SensorDataPrediction
    {
        //vector to hold alert,score,p-value values
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }

    class IMU
    {

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "acceleration_only.csv");

        public static long? _docsize = 0;
        private static System.Timers.Timer aTimer;
        private static List<double> corrected = new List<double>();
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<SensorData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            var accelerations = mlContext.Data.CreateEnumerable<SensorData>(dataView, reuseRowObject: false).ToList();                
            var filter = new KalmanFilter();
            List<double> measurements = new List<double>();
            List<double> states = new List<double>();
            Random rnd = new Random();

            for (int k = 0; k < accelerations.Count; k++)
            {
                measurements.Add(accelerations[k].Acceleration_x);
                filter.Update(new[] { accelerations[k].Acceleration_x });
                states.Add(filter.getState()[0]);
            }

            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("data.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, states.Count,                          // x-axis range
                 states.Min(x => x), states.Max(x => x),                         // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Poll",                         // x-axis label
                "Acceleration",                        // y-axis label
                "acc-x");     // plot title
            pl.line(
                (from x in Enumerable.Range(0, states.Count()) select (double)x).ToArray(),
                (from p in states select (double)p).ToArray()
            );

            pl.eop();
        }
        public class KalmanFilter
        {
            private int L;
            private int m;
            /// sigma-points dispersion around mean
            private double alpha;
            private double ki;
            ///type of distribution
            private double beta;
            private double lambda;
            private double c;
            /// mean weights
            private Matrix<double> Wm;
            /// Covariance weights
            private Matrix<double> Wc;
            /// State
            private Matrix<double> x;
            /// Covariance
            private Matrix<double> P;
            /// Std of process 
            private double q;
            /// Std of measurement 
            private double r;
            /// Covariance of process
            private Matrix<double> Q;
            /// Covariance of measurement 
            private Matrix<double> R;
            public KalmanFilter(int L = 0)
            {
                this.L = L;
            }

            private void f(int w = 0)
            {
                q = 0.02;
                r = 0.3;
                x = q * Matrix.Build.Random(L, 1); 
                P = Matrix.Build.Diagonal(L, L, 1); //initial state covraiance

                Q = Matrix.Build.Diagonal(L, L, q * q); //covariance of process
                R = Matrix.Build.Dense(m, m, r * r); //covariance of measurement  

                alpha = 1e-3f;
                ki = 0;
                beta = 2f;
                lambda = alpha * alpha * (L + ki) - L;
                c = L + lambda;

                //weights for means
                Wm = Matrix.Build.Dense(1, (2 * L + 1), 0.5 / c);
                Wm[0, 0] = lambda / c;

                //weights for covariance
                Wc = Matrix.Build.Dense(1, (2 * L + 1));
                Wm.CopyTo(Wc);
                Wc[0, 0] = Wm[0, 0] + 1 - alpha * alpha + beta;

                c = Math.Sqrt(c);
            }

            public void Update(double[] measurements)
            {
                if (m == 0)
                {
                    var mNum = measurements.Length;
                    if (mNum > 0)
                    {
                        m = mNum;
                        if (L == 0) L = mNum;
                        f();
                    }
                }

                var z = Matrix.Build.Dense(m, 1, 0);
                z.SetColumn(0, measurements);

                //sigma points around x
                Matrix<double> X = GetSigmaPoints(x, P, c);

                //unscented transformation of process
                // X1=sigmas(x1,P1,c) - sigma points around x1
                //X2=X1-x1(:,ones(1,size(X1,2))) - deviation of X1
                Matrix<double>[] ut_f_matrices = UnscentedTransform(X, Wm, Wc, L, Q);
                Matrix<double> x1 = ut_f_matrices[0];
                Matrix<double> X1 = ut_f_matrices[1];
                Matrix<double> P1 = ut_f_matrices[2];
                Matrix<double> X2 = ut_f_matrices[3];

                //unscented transformation of measurments
                Matrix<double>[] ut_h_matrices = UnscentedTransform(X1, Wm, Wc, m, R);
                Matrix<double> z1 = ut_h_matrices[0];
                Matrix<double> Z1 = ut_h_matrices[1];
                Matrix<double> P2 = ut_h_matrices[2];
                Matrix<double> Z2 = ut_h_matrices[3];

                //transformed cross-covariance
                Matrix<double> P12 = (X2.Multiply(Matrix.Build.Diagonal(Wc.Row(0).ToArray()))).Multiply(Z2.Transpose());

                Matrix<double> K = P12.Multiply(P2.Inverse());

                //state update
                x = x1.Add(K.Multiply(z.Subtract(z1)));
                //covariance update 
                P = P1.Subtract(K.Multiply(P12.Transpose()));
            }

            public double[] getState()
            {
                return x.ToColumnArrays()[0];
            }

            public double[,] getCovariance()
            {
                return P.ToArray();
            }

            /// <summary>
            /// Transformation
            /// </summary>
            /// <param name="f">nonlinear map</param>
            /// <param name="X">sigma points</param>
            /// <param name="Wm">Weights for means</param>
            /// <param name="Wc">Weights for covariance</param>
            /// <param name="n">numer of outputs of f</param>
            /// <param name="R">additive covariance</param>
            /// <returns>[transformed mean, transformed smapling points, transformed covariance, transformed deviations</returns>
            private Matrix<double>[] UnscentedTransform(Matrix<double> X, Matrix<double> Wm, Matrix<double> Wc, int n, Matrix<double> R)
            {
                int L = X.ColumnCount;
                Matrix<double> y = Matrix.Build.Dense(n, 1, 0);
                Matrix<double> Y = Matrix.Build.Dense(n, L, 0);

                Matrix<double> row_in_X;
                for (int k = 0; k < L; k++)
                {
                    row_in_X = X.SubMatrix(0, X.RowCount, k, 1);
                    Y.SetSubMatrix(0, Y.RowCount, k, 1, row_in_X);
                    y = y.Add(Y.SubMatrix(0, Y.RowCount, k, 1).Multiply(Wm[0, k]));
                }

                Matrix<double> Y1 = Y.Subtract(y.Multiply(Matrix.Build.Dense(1, L, 1)));
                Matrix<double> P = Y1.Multiply(Matrix.Build.Diagonal(Wc.Row(0).ToArray()));
                P = P.Multiply(Y1.Transpose());
                P = P.Add(R);

                Matrix<double>[] output = { y, Y, P, Y1 };
                return output;
            }

            /// <summary>
            /// Sigma points around reference point
            /// </summary>
            /// <param name="x">reference point</param>
            /// <param name="P">covariance</param>
            /// <param name="c">coefficient</param>
            /// <returns>Sigma points</returns>
            private Matrix<double> GetSigmaPoints(Matrix<double> x, Matrix<double> P, double c)
            {
                Matrix<double> A = P.Cholesky().Factor;

                A = A.Multiply(c);
                A = A.Transpose();

                int n = x.RowCount;

                Matrix<double> Y = Matrix.Build.Dense(n, n, 1);
                for (int j = 0; j < n; j++)
                {
                    Y.SetSubMatrix(0, n, j, 1, x);
                }

                Matrix<double> X = Matrix.Build.Dense(n, (2 * n + 1));
                X.SetSubMatrix(0, n, 0, 1, x);

                Matrix<double> Y_plus_A = Y.Add(A);
                X.SetSubMatrix(0, n, 1, n, Y_plus_A);

                Matrix<double> Y_minus_A = Y.Subtract(A);
                X.SetSubMatrix(0, n, n + 1, n, Y_minus_A);

                return X;
            }
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
