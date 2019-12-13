using Iot.Device.CpuTemperature;
using System;

namespace ActTracker
{
    class Gyroscope
    {
        static CpuTemperature temp = new CpuTemperature();
        static void Main(string[] args)
        {
            while (true)
            {
                if (temp.IsAvailable)
                {
                    Console.WriteLine($"The cpu temp is {temp.Temperature.Celsius}");
                }
                else
                {
                    Console.WriteLine("Temp not available");
                }
            }
        }
    }
}
