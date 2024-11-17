package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class FilterDisplay {

    /**
     * Takes a model and saves the filters from the CNN to the specified path.
     * @param model the model to analyze
     * @param layerIndex the layer whose filters you want to look at
     * @param filterIndex
     * @throws IOException
     */
// Method to save a specific filter as an image
    public static void saveFilterAsImage(MultiLayerNetwork model, int layerIndex, int filterIndex, String outputPath) throws IOException {
        // Get the weights (filters) from the specified layer
        Layer layer = model.getLayer(layerIndex);
        INDArray filterWeights = layer.getParam("W");  // Get the weights of the layer (filters)

        // Retrieve the specific filter (kernel) from the filterWeights
        // The shape is [numFilters, numChannels, filterHeight, filterWidth, depth]
        // Extract the filter of interest (filterIndex)
        INDArray filter = filterWeights.get(NDArrayIndex.point(filterIndex), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

        // Normalize the filter values to the range [0, 255] for display
        INDArray normalizedFilter = normalizeToImage(filter);

        // Convert the normalized filter to a BufferedImage
        BufferedImage image = toBufferedImage(normalizedFilter);

        // Save the BufferedImage to a file using ImageIO
        ImageIO.write(image, "PNG", new File(outputPath));
        System.out.println("Filter saved to: " + outputPath);
    }

    // Normalize the filter values to the range [0, 255] for visualization
    private static INDArray normalizeToImage(INDArray filter) {
        // Find the min and max of the filter to scale the values between 0 and 255
        double min = filter.minNumber().doubleValue();
        double max = filter.maxNumber().doubleValue();

        // Apply the normalization: (value - min) / (max - min) * 255
        INDArray normalized = filter.sub(min).div(max - min).mul(255);
        return normalized;
    }

    // Convert the normalized INDArray to a BufferedImage
    private static BufferedImage toBufferedImage(INDArray filter) {
        long numChannels = filter.size(1); // 3 channels (RGB)
        long filterHeight = filter.size(2); // Height of the filter
        long filterWidth = filter.size(3); // Width of the filter

        // Create a BufferedImage with RGB type
        BufferedImage image = new BufferedImage((int) filterWidth, (int) filterHeight, BufferedImage.TYPE_INT_RGB);

        // Loop through the filter and set the pixel values to the BufferedImage
        for (int y = 0; y < filterHeight; y++) {
            for (int x = 0; x < filterWidth; x++) {
                // Get the RGB values for the pixel at position (x, y)
                int red = (int) filter.getDouble(0, 0, y, x);   // Red channel (for RGB)
                int green = (int) filter.getDouble(0, 1, y, x); // Green channel (for RGB)
                int blue = (int) filter.getDouble(0, 2, y, x);  // Blue channel (for RGB)

                // Clamp the pixel values to be between 0 and 255
                red = Math.min(Math.max(red, 0), 255);
                green = Math.min(Math.max(green, 0), 255);
                blue = Math.min(Math.max(blue, 0), 255);

                // Set the pixel color for the BufferedImage
                int rgb = new Color(red, green, blue).getRGB();
                image.setRGB(x, y, rgb);
            }
        }

        return image;
    }
}
