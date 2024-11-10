package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.StateDatum;
import com.opencsv.CSVWriter;
import org.bytedeco.opencv.presets.opencv_core;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getBoardIndex;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

public abstract class NetworkTrainer {
    protected static final Logger log = LoggerFactory.getLogger(ValueNetworkTrainer.class);
    protected static PrintStream outStream;

    abstract public void train(String baseDir);

    /**
     * takes in raw training data and creates a bunch of images that represent it, packs it all into a dataset and saves it. Running on 650 games or so this took a few seconds and the saved data was around 1.9GB
     * @param dataDumpDir the directory to save the images.
     * @throws FileNotFoundException if unable to concatenate data files
     */
    protected static void preprocessPolicyImageInput(String dataDumpDir) throws IOException {
        DataSet dataFromFile = concatenateValueDataFiles(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\"), "series1", "policy");
        FileWriter fileWriter = new FileWriter(dataDumpDir + "labels.csv");
        CSVWriter csvWriter = new CSVWriter(fileWriter);
        int size = (int) dataFromFile.getLabels().shape()[1] + 1;
        String[] headers = new String[size];
        for (int j = 0; j < size; j++) {
            if (j == 0) {
                headers[j] = "ID";
            }else {
                headers[j] = "X" + j;
            }
        }
        csvWriter.writeNext(headers);
        for (int i = 0; i < dataFromFile.getFeatures().shape()[0]; i++) {
            BufferedImage image =  Resources.getImageForAI(BoardState.getBoardStateFromFlatVector(dataFromFile.getFeatures().getRow(i).toDoubleVector(), 3), 1);
            try {
                ImageIO.write(image, "PNG", new File(dataDumpDir + i + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }
            double[] labels = dataFromFile.getLabels().getRow(i).toDoubleVector();
            String[] strs = new String[labels.length + 1];
            strs[0] = String.valueOf(i);
            for (int j = 1; j < labels.length + 1; j++) {
                strs[j] = String.valueOf(labels[j - 1]);
            }
            csvWriter.writeNext(strs);

        }

        csvWriter.close();

    }


    protected static DataSet getInputDataSetFromRawFilepath(String filepath, String seriesName, String prefix) throws FileNotFoundException {
        DataSet rawData = concatenateValueDataFiles(new File(filepath),seriesName, prefix);
        double [][][][][] inputVectors = new double[rawData.getFeatures().rows()][][][][];
        double[][] rawInputMatrix = rawData.getFeatures().toDoubleMatrix();
        for (int i = 0; i < rawData.getFeatures().rows(); i++) {
            inputVectors[i] = getBoardTensor(rawInputMatrix[i]);
        }
        int dataRows = rawData.getFeatures().rows();
        int inputChannels = 4;//num channels
        int boardSize = 3;//the other four dimensions are all boardsize, minor and major rows and colummns.


        double[] flatInputVectors = new double[dataRows  * boardSize * boardSize * boardSize * boardSize * inputChannels];
        int index = 0;
        for (int i = 0; i < dataRows; i++) {
            for (int j = 0; j < boardSize * boardSize; j++) {
                for (int k = 0; k < boardSize; k++) {
                    for (int l = 0; l < boardSize; l++) {
                        for (int m = 0; m < inputChannels; m++) {
                            flatInputVectors[index++] = inputVectors[i][j][k][l][m];
                        }
                    }
                }
            }
        }
        INDArray inputFeatures = Nd4j.create(flatInputVectors, new int[]{dataRows, boardSize * boardSize, boardSize, boardSize,inputChannels});
        return new DataSet(inputFeatures, rawData.getLabels());
    }



    /**
     * This method converts a densely stored board state into a tensor form that can be easily understood by the model
     *
     * <p>
     * <p>
     * This data structure needs to work with a 3D dimensional convolutional layer.
     * therefore the input must be 5D.
     * the shape of the input is as follows:
     * [batch or datapoint index, major index,minor row,minor col, channel]
     * the major indexes range from 0 to 8 and follow the standard top left to bottom right scheme that the data arrays use to store the data.
     * the value is 0 or 1 based on the channel
     *
     * Channel 1: "X" presence (1 for "X", 0 otherwise)
     * Channel 2: "O" presence (1 for "O", 0 otherwise)
     * Channel 3: Empty presence (1 if empty, 0 otherwise)
     * Channel 4: Active state (1 if active, 0 if inactive)
     * for board activity all entries for the activity channel for that subboard will be hot.
     * <p>
     * The input is 90 doubles the first 81 are either 1, -1, or 0 representing if X O or nobody occupies that specific square. The order of indexing is for each sub board top left to top right
     * then the next row left to right, and the third row. The order of indexing of the major boards is the same as that of the minor boards.
     *
     * @param dataArray the array of data to convert
     * @return the tensor representing the {@link com.hottes.caleb.ultimateticktacktoe.BoardState}
     * @see StateDatum
     * @see BoardState#getBoardStateFlatVector()
     */
    protected static double[][][][] getBoardTensor(double[] dataArray) {
        int boardSize = 3;
        double[][][][] tensor = new double[boardSize * boardSize][boardSize][boardSize][4];
        for (int majRow = 0; majRow < boardSize; majRow++) {
            for (int majCol = 0; majCol < boardSize; majCol++) {
                float activity = dataArray[Resources.getActivityIndex(majRow, majCol, boardSize)] == 1?1:0;
                for (int minRow = 0; minRow < boardSize; minRow++) {
                    for (int minCol = 0; minCol < boardSize; minCol++) {
                        double val = dataArray[Resources.getIndex(majRow, majCol, minRow, minCol, boardSize)];//1 if X, 0 if empty, -1 if O
                        tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][1] = 0;//initially not "O" but if we determine that it is "O" then we will set it later.
                        tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][3] = activity;//set the activity channel
                        if (val == 0) {//empty channel first, most likly
                            tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][2] = 1;
                        } else {
                            tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][2] = 0;
                            if (val == 1) {
                                tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][0] = 1;
                            } else {
                                tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][0] = 0;
                                //so its not empty, and its not X so its O
                                tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][1] = 1;
                            }//end not X
                        }//end not empty
                    }

                }

            }

        }
        return tensor;
    }//end method


    /**
     * takes in a dataset of any rank and splits it into training and testing datasets at the given percentage.
     * The only reason I am making this method is because the built in one doesn't support datasets above rank 4 for some reason.
     * this operates the same as the one in the source code does, except it only works for rank 5
     * @param dataSet the dataset to split
     * @param percentTrain the ratio to split the data
     * @return a split test train object.
     */
    public static SplitTestAndTrain splitTestAndTrain(DataSet dataSet, double percentTrain) {
        if (percentTrain >= 1 || percentTrain <= 0) {
            throw new IllegalArgumentException("Percent train must be between 0 and 1 exclusive");
        }
        if (dataSet.getFeatures().rank() != 5) {
            throw new IllegalArgumentException("This method is for rank 5 splitting only");
        }
        long[] shape = dataSet.getFeatures().shape();
        int numTrainingRows = (int) Math.ceil(shape[0] * percentTrain);
        long numRows = shape[0];

        DataSet train = new DataSet();
        train.setFeatures(dataSet.getFeatures().get(interval(0, numTrainingRows), all(), all(), all(), all()));
        train.setLabels(dataSet.getLabels().get(interval(0, numTrainingRows), all()));
        DataSet test = new DataSet();
        test.setFeatures(dataSet.getFeatures().get(interval(numTrainingRows, numRows), all(), all(), all(), all()));
        test.setLabels(dataSet.getLabels().get(interval(numTrainingRows, numRows), all()));
        return new SplitTestAndTrain(train, test);
    }


    protected static  void log(String message) {
        log.info(message);
        outStream.println(message);

    }

    /**
     * Takes a filepath and series name and puts all data files with a specific prefix eg value or policy
     *
     * @param dataDir    the source of the data storage folder
     * @param seriesName the name of the data series
     * @return the dataset
     */
    public static DataSet concatenateValueDataFiles(File dataDir, String seriesName, String prefix) throws FileNotFoundException {
        if (!dataDir.isDirectory()) {
            throw new IllegalArgumentException("Provided file is not a folder. ");
        }
        File seriesDir = new File(dataDir.getAbsolutePath() + "/" + seriesName);
        if (!seriesDir.isDirectory()) {
            throw new FileNotFoundException("Data folder for series: " + seriesName + " not found.");
        }
        List<DataSet> dataSets = new ArrayList<>();
        for (File file : Objects.requireNonNull(seriesDir.listFiles((dir, name) -> name.endsWith(".bin") && name.startsWith(prefix)))) {
            DataSet dataSet = new DataSet();
            dataSet.load(file);
            dataSets.add(dataSet);
        }
        return concatenateDataSets(dataSets);

    }


    private static DataSet concatenateDataSets(List<DataSet> dataSets) {
        // Get total number of features and labels
        INDArray features = dataSets.get(0).getFeatures().castTo(DataType.DOUBLE);
        INDArray labels = dataSets.get(0).getLabels().castTo(DataType.DOUBLE);

        for (int i = 1; i < dataSets.size(); i++) {
            ;
            features = Nd4j.vstack(features, dataSets.get(i).getFeatures().castTo(DataType.DOUBLE));
            labels = Nd4j.vstack(labels, dataSets.get(i).getLabels().castTo(DataType.DOUBLE));
        }

        return new DataSet(features, labels);
    }





}
