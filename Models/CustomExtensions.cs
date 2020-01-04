using System;
using System.Collections.Generic;
using System.Text;

namespace ActTracker.Models
{
    public class CustomBool
    {
        public double index { get; set; }
        public bool Max { get; set; }
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
    class ThreePoints
    {
        public List<CustomDouble> three_points { get; set; }
        public double average { get; set; }
    }
}
