using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Globalization;

namespace ActTracker.KalmanHelper
{
    public class QuaternionData
    {
        private float[] quaternion;
        public float[] Quaternion
        {
            get
            {
                return quaternion;
            }
            set
            {
                if (value.Length != 4)
                {
                    throw new Exception("Quaternion vector must be of 4 elements");
                }
                float norm = (float)Math.Sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2] + value[3] * value[3]);
                quaternion = value;
                quaternion[0] /= norm;
                quaternion[1] /= norm;
                quaternion[2] /= norm;
                quaternion[3] /= norm;
            }
        }

        public QuaternionData()
            : this(new float[] { 1, 0, 0, 0 })
        {
        }

        public QuaternionData(float[] quaternion)
        {
            Quaternion = quaternion;
        }

        public QuaternionData ConvertToConjugate()
        {
            return new QuaternionData(new float[] { Quaternion[0], -Quaternion[1], -Quaternion[2], -Quaternion[3] });
        }

        public float[] ConvertToRotationMatrix()
        {
            float R11 = 2 * Quaternion[0] * Quaternion[0] - 1 + 2 * Quaternion[1] * Quaternion[1];
            float R12 = 2 * (Quaternion[1] * Quaternion[2] + Quaternion[0] * Quaternion[3]);
            float R13 = 2 * (Quaternion[1] * Quaternion[3] - Quaternion[0] * Quaternion[2]);
            float R21 = 2 * (Quaternion[1] * Quaternion[2] - Quaternion[0] * Quaternion[3]);
            float R22 = 2 * Quaternion[0] * Quaternion[0] - 1 + 2 * Quaternion[2] * Quaternion[2];
            float R23 = 2 * (Quaternion[2] * Quaternion[3] + Quaternion[0] * Quaternion[1]);
            float R31 = 2 * (Quaternion[1] * Quaternion[3] + Quaternion[0] * Quaternion[2]);
            float R32 = 2 * (Quaternion[2] * Quaternion[3] - Quaternion[0] * Quaternion[1]);
            float R33 = 2 * Quaternion[0] * Quaternion[0] - 1 + 2 * Quaternion[3] * Quaternion[3];
            return new float[] { R11, R12, R13,
                                 R21, R22, R23,
                                 R31, R32, R33 };
        }

        public float[] ConvertToEulerAngles()
        {
            float phi = (float)Math.Atan2(2 * (Quaternion[2] * Quaternion[3] - Quaternion[0] * Quaternion[1]), 2 * Quaternion[0] * Quaternion[0] - 1 + 2 * Quaternion[3] * Quaternion[3]);
            float theta = (float)-Math.Atan((2.0 * (Quaternion[1] * Quaternion[3] + Quaternion[0] * Quaternion[2])) / Math.Sqrt(1.0 - Math.Pow((2.0 * Quaternion[1] * Quaternion[3] + 2.0 * Quaternion[0] * Quaternion[2]), 2.0)));
            float psi = (float)Math.Atan2(2 * (Quaternion[1] * Quaternion[2] - Quaternion[0] * Quaternion[3]), 2 * Quaternion[0] * Quaternion[0] - 1 + 2 * Quaternion[1] * Quaternion[1]);
            return new float[] { Rad2Deg(phi), Rad2Deg(theta), Rad2Deg(psi) };
        }
        private float Rad2Deg(float radians)
        {
            return 57.2957795130823f * radians;
        }
        public string ConvertToEulerAnglesCSVstring()
        {
            float[] euler = ConvertToEulerAngles();
            return euler[0].ToString(CultureInfo.InvariantCulture) + "," + euler[1].ToString(CultureInfo.InvariantCulture) + "," + euler[2].ToString(CultureInfo.InvariantCulture);
        }
    }
}