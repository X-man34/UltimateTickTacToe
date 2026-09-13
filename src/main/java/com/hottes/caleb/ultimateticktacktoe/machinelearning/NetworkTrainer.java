package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.StateDatum;
import com.opencsv.CSVWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
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

    /**
     * Trains a neural network according to whatever the dev has in mind.
     * This method is more of a script that is changed as needed during development to get stuff done
     * @param baseDir the AI directory, should have whatever files are needed as determined through the ever changing implementation
     */
    abstract public void train(String baseDir);

    /**
     * takes in raw training data and creates a bunch of images that represent it, packs it all into a dataset and saves it. Running on 650 games or so this took a few seconds and the saved data was around 1.9GB
     * @param dataDumpDir the directory to save the images.
     * @throws FileNotFoundException if unable to concatenate data files
     */
    protected static void preprocessPolicyImageInput(String dataDumpDir, String baseDataDir, String seriesName, double trainTestRatio, int boardSize, int scaleFactor) throws IOException {
        DataSet dataFromFile = concatenateDataFiles(new File(baseDataDir), seriesName, "policy");
        SplitTestAndTrain testAndTrain = dataFromFile.splitTestAndTrain(trainTestRatio);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        writeDataSet(trainingData, dataDumpDir + "\\train\\", boardSize, scaleFactor);
        writeDataSet(testData, dataDumpDir + "\\test\\", boardSize, scaleFactor);


    }

    /**
     * This method takes a raw standard format policy dataset, ie one from concatenated data files as well as a directory and creates policy network traning input.
     * Its not creating data but its reformatting the board state tensors into images and saving them. This method also leaves a labels.csv file it the directory containing the label for each images.
     * the ID column in the csv file corresponds to the filename of each image. This method is an abstraction so train and test
     * data sets can easily be constructed from raw data.
     * if there is an IO exception when writing an image it is ignored and that image (and label) are skipped.
     *The data shuffled before being processed.
     * @param dataset the standard format policy dataset
     * @param dataDumpDir the directory to place the images and label file
     * @param boardSize the size of the board, probably won't work unless its 3
     * @param scaleFactor determines the size of the images. a factor of one results in 76x76 images
     * @throws IOException if some of the large amount of file IO goes wrong.
     */
    private static void writeDataSet(DataSet dataset, String dataDumpDir, int boardSize, int scaleFactor) throws IOException {
        File saveDir = new File(dataDumpDir);
        saveDir.mkdirs();
        File labelFile = new File(dataDumpDir + "labels.csv");
        if (!labelFile.createNewFile()) {
            labelFile.createNewFile();
        }

        FileWriter fileWriter = new FileWriter(labelFile);
        CSVWriter csvWriter = new CSVWriter(fileWriter);
        int size = (int) dataset.getLabels().shape()[1] + 1;
        String[] headers = new String[size];
        for (int j = 0; j < size; j++) {
            if (j == 0) {
                headers[j] = "ID";
            }else {
                headers[j] = "X" + j;
            }
        }
        csvWriter.writeNext(headers);
        dataset.shuffle();
        for (int i = 0; i < dataset.getFeatures().shape()[0]; i++) {
            BufferedImage image =  Resources.getImageForAI(BoardState.getBoardStateFromFlatVector(dataset.getFeatures().getRow(i).toDoubleVector(), boardSize), scaleFactor);
            try {
                ImageIO.write(image, "PNG", new File(dataDumpDir + i + ".png"));
            } catch (IOException e) {
                continue;
            }
            double[] labels = dataset.getLabels().getRow(i).toDoubleVector();
            String[] strs = new String[labels.length + 1];
            strs[0] = String.valueOf(i);
            for (int j = 1; j < labels.length + 1; j++) {
                strs[j] = String.valueOf(labels[j - 1]);
            }
            csvWriter.writeNext(strs);

        }

        csvWriter.close();
    }


    /**
     * This method takes in a filepath to a directory of saved standard data files. It combines them all together and converts the features of the data into
     * one hot encoded tensor format. This data is then returned, the labels are left alone. (These docs are being written after the fact and there are no useages) it appears that
     * this was used for value and policy networks. I think i'll deprecate it because networks made this way didn't perform well.
     * @param filepath the standard data directory path
     * @param seriesName the series name
     * @param prefix the data prefix, either "value" or "policy"
     * @return a converted dataset
     * @throws FileNotFoundException if fileIO goes bad :(
     */
    @Deprecated
    protected static DataSet getInputDataSetFromRawFilepath(String filepath, String seriesName, String prefix) throws FileNotFoundException {
        DataSet rawData = concatenateDataFiles(new File(filepath),seriesName, prefix);
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


    /**
     * logs info both with the logger and also to whatever the outSteam variable points to
     * This method looks like an excuse for not using logger config files
     * @param message the message to log
     */
    protected static  void log(String message) {
        log.info(message);
        outStream.println(message);

    }

    /**
     * Takes a filepath and series name and puts all data files with a specific prefix eg value or policy
     *together into one data file.
     * @param dataDir    the source of the data storage folder
     * @param seriesName the name of the data series
     * @return the dataset
     */
    public static DataSet concatenateDataFiles(File dataDir, String seriesName, String prefix) throws FileNotFoundException {
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


    /**
     * preforms the dirty work of {@link NetworkTrainer#concatenateDataFiles(File, String, String)}
     * @param dataSets the datasets to concatenate
     * @return the concatenated dataset
     */
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
