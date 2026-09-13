package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.opencsv.CSVIterator;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import javafx.util.Pair;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.io.*;
import java.rmi.UnexpectedException;
import java.util.*;

public class ImageRegressionDataSetIterator implements DataSetIterator {

    private final int height;
    private final int width;
    private final int channels;
    private final File imageDir;
    private final int batchSize;
    private final File labelFile;
    private final String extension;
    private Iterator<File> imageFileIterator;
    private final ArrayList<Pair<Integer, List<Double>>> labelData;
    private DataSetPreProcessor preProcessor;

    public ImageRegressionDataSetIterator(int width, int height, int channels, File imageDir, int batchSize, String fileExtension, DataSetPreProcessor preProcessor) throws FileNotFoundException {
        this(width, height, channels, imageDir, batchSize, fileExtension);
        setPreProcessor(preProcessor);
    }

    public ImageRegressionDataSetIterator(int width, int height, int channels, File imageDir, int batchSize, String fileExtension) throws FileNotFoundException {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.extension = fileExtension;
        if (!imageDir.isDirectory()) {
            throw new IllegalArgumentException("Image directory is not a directory");
        }else {
            this.imageDir = imageDir;

        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        this.batchSize = batchSize;
        this.labelFile = new File(imageDir.getAbsolutePath() + "\\labels.csv");
        if (!labelFile.exists()) {
            throw new FileNotFoundException("Could not find label file: " + labelFile.getAbsolutePath());
        }
        imageFileIterator = getFileIterator(imageDir, List.of(extension));
        labelData = loadLabels(labelFile);

    }

    /**
     * reutrns a row vector of the label associated with this specific imageID
     * @param imageID
     * @return
     */
    public INDArray findLabel(int imageID) {
        Optional<Pair<Integer, List<Double>>> potentialLabel =  labelData.stream().filter(integerListPair -> integerListPair.getKey() == imageID).findFirst();
        if (potentialLabel.isPresent()) {
            List<Double> label = potentialLabel.get().getValue();
            return Nd4j.create(label);
        }else {
            throw new RuntimeException("unable to find label");
        }


    }

    public static ArrayList<Pair<Integer, List<Double>>> loadLabels(File labelFile) {
        try {
            CSVIterator csvIterator = new CSVIterator(new CSVReader(new BufferedReader(new FileReader(labelFile))));
            if (!csvIterator.hasNext()) {
                throw new UnexpectedException("Why does the CSV file not have at least one row?");
            }
            csvIterator.next();//consume the title row and assume the first value is the ID and is the same as the filename of the image is corresponds to.
            ArrayList<Pair<Integer, List<Double>>> labels = new ArrayList<>();
            while (csvIterator.hasNext()) {

                List<String> data = Arrays.stream(csvIterator.next()).toList();
                List<Double> doubleData = new ArrayList<>();
                data.forEach(s -> doubleData.add(Double.parseDouble(s)));
                labels.add(new Pair<>(doubleData.getFirst().intValue(), doubleData.stream().skip(1).toList()));
            }
            return labels;
        } catch (IOException | CsvValidationException e) {
            throw new RuntimeException(e);
        }
    }


    private Iterator<File> getFileIterator(File imageDir, List<String> extensions) throws FileNotFoundException {
        if (!imageDir.isDirectory()) {
            throw new IllegalArgumentException();
        }
        File[] files = imageDir.listFiles();

        if (files != null) {
            List<File> fileList = new ArrayList<>();
            for (File file : files) {
                if (file.isFile()) {
                    if (extensions.contains(file.getName().split("\\.")[1].toLowerCase())) {
                        fileList.add(file);
                    }
                }
            }
            return fileList.iterator();
        } else {
             throw new FileNotFoundException("No file found :(");
        }
    }


    /**
     * returns a dataset with the number of examples equal to the batch size
     * @return the next dataset.
     */
    @Override
    public DataSet next() {
        return next(batchSize);
    }

    /**
     * This method loads the specifed number of images and their labels into a dataset and returns it.
     * If there are some but not enough images to finish the batch then as many as are left are loaded.
     * If errors are ecountered the image what was attempted to be loaded is skipped
     * if the iterator does not have any images it will return null, use {@link ImageRegressionDataSetIterator#hasNext()} to avoid this
     * @param num the number of examples
     * @return a dataset of num datapoint.
     */
    @Override
    public DataSet next(int num) {
        if (!hasNext()) {
           return null;
        }
        int imagesLoaded = 0;
        INDArray featuresArray = Nd4j.create(num, channels, width, height);
        INDArray labelsArray = Nd4j.create(num, 81);
        long startTime = System.currentTimeMillis();
        while (imageFileIterator.hasNext()) {
            try {
                File imageFile = imageFileIterator.next();
                INDArray matrix = loadImage(imageFile);
                featuresArray.put(new INDArrayIndex[]{NDArrayIndex.point(imagesLoaded), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, matrix);
                labelsArray.put(new INDArrayIndex[]{NDArrayIndex.point(imagesLoaded), NDArrayIndex.all()}, findLabel(Integer.parseInt(imageFile.getName().split("\\.")[0])));
                imagesLoaded++;
            } catch (IOException e) {
                //oh well, maybe the next image will work
            }
            if (imagesLoaded >= num) {
                break;
            }
            if (!imageFileIterator.hasNext()) {
                System.out.println();
            }
        }
        //System.out.println("Took: " + (System.currentTimeMillis() - startTime) + " ms to load " + imagesLoaded + " images");
        // Trim the features and labels arrays to only the rows that have been populated (based on imagesLoaded)
        INDArray featuresTrimmed = featuresArray.get(NDArrayIndex.interval(0, imagesLoaded), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray labelsTrimmed = labelsArray.get(NDArrayIndex.interval(0, imagesLoaded), NDArrayIndex.all());

        // Create the DataSet with trimmed arrays
        DataSet dataSet = new DataSet(featuresTrimmed, labelsTrimmed);

        if (preProcessor != null) {
            preProcessor.preProcess(dataSet);
        }
        return dataSet;
    }

    @Override
    public int inputColumns() {
        return -1;//Not applicable, input data rank is 4, so columns don't really mean anything.
    }

    @Override
    public int totalOutcomes() {
        return labelData.getFirst().getValue().size();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        try {
            imageFileIterator = getFileIterator(imageDir, List.of(extension));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return imageFileIterator.hasNext();
    }


    private INDArray loadImage(File imageFile) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        return loader.asMatrix(imageFile);
    }

    public INDArray getFirstNLabels(int n) {
        INDArray output = Nd4j.create(n, labelData.get(0).getValue().size());

        for (int i = 0; i < n; i++) {
            Pair<Integer, List<Double>> label = labelData.get(i);
            double[] labelArray = label.getValue().stream().mapToDouble(Double::doubleValue).toArray();
            INDArray labels = Nd4j.create(labelArray);
            output.putRow(label.getKey(), labels);
        }

        return output;
    }

    public INDArray getAllLabels() {
        return  getFirstNLabels(labelData.size());
    }



}
