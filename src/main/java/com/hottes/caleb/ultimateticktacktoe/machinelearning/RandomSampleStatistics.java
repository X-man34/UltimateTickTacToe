package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import org.bytedeco.opencv.presets.opencv_core;

import java.text.DecimalFormat;

public record RandomSampleStatistics(double mean, double marginOfError, double confidence, double standardDeviation, String units) {

    @Override
    public String toString() {
        DecimalFormat format = new DecimalFormat(".##");
        DecimalFormat percentFormat = new DecimalFormat("##");
        return "Mean is: " + format.format(mean) + " " + units + " +/- " + format.format(marginOfError) + " (" + percentFormat.format(confidence * 100) + " %) and has standard deviation " + format.format(standardDeviation) + " " +  units + ".";
    }
}
