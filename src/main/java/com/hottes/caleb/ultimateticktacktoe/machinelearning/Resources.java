package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Resources {


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

    public static int getActivityIndex(int majRow, int majCol, int boardSize) {
        return getIndex(boardSize - 1, boardSize - 1, boardSize - 1, boardSize - 1, boardSize) + 1 + (majRow * boardSize) + majCol;
    }

    /**
     * takes in information about a ultimate tick tac toe square and converts it to a linear index.
     * this is so the entire board can be represented as a linear array for vectorization.
     * this is for machine learning
     *
     * @param majRow      the index of the major row
     * @param majCol      the index of the major column
     * @param minRow      the index of the minor row
     * @param minCol the index of the minor column
     * @return the linear index of this spot.
     */
    public static int getIndex(int majRow, int majCol, int minRow, int minCol, int boardSize) {
        int majorIndex = majRow * boardSize + majCol;
        int minorIndex = minRow * boardSize + minCol;
        return majorIndex * boardSize * boardSize + minorIndex;
    }

    /**
     * takes in a board state and returns the input for the value network version 1.0
     * while {@link ValueNetworkTrainer#getBoardTensor(double[])} is liable to change based on what is currently needed, this method is intended to
     * serialize that code so old versions of the model can be used.
     * From {@link ValueNetworkTrainer#getBoardTensor(double[])}
     *      * This data structure needs to work with a 3D dimensional convolutional layer.
     *      * therefore the input must be 5D.
     *      * the shape of the input is as follows:
     *      * [batch or datapoint index, major index,minor row,minor col, channel]
     *      * the major indexes range from 0 to 8 and follow the standard top left to bottom right scheme that the data arrays use to store the data.
     *      * the value is 0 or 1 based on the channel
     *      *
     *      * Channel 1: "X" presence (1 for "X", 0 otherwise)
     *      * Channel 2: "O" presence (1 for "O", 0 otherwise)
     *      * Channel 3: Empty presence (1 if empty, 0 otherwise)
     *      * Channel 4: Active state (1 if active, 0 if inactive)
     *      * for board activity all entries for the activity channel for that subboard will be hot.
     *      * <p>
     *      * The input is 90 doubles the first 81 are either 1, -1, or 0 representing if X O or nobody occupies that specific square. The order of indexing is for each sub board top left to top right
     *      * then the next row left to right, and the third row. The order of indexing of the major boards is the same as that of the minor boards.
     * @param boardState the state we need input from
     * @return an INDArray that can be fed directly into the model of the correct version
     */
    public static INDArray getValueNetworkInputV1_0(BoardState boardState) {
        int boardSize = boardState.getBoardSize();
        double[][][][] tensor = new double[boardSize * boardSize][boardSize][boardSize][4];
        for (int majRow = 0; majRow < boardSize; majRow++) {
            for (int majCol = 0; majCol < boardSize; majCol++) {
                float activity =  boardState.getMinorBoardAt(majRow, majCol).isActive()?1:0;
                for (int minRow = 0; minRow < boardSize; minRow++) {
                    for (int minCol = 0; minCol < boardSize; minCol++) {
                        double val = boardState.getMinorBoardAt(majRow, majCol).itemAt(minRow, minCol);//1 if X, 0 if empty, -1 if O
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
        double[] flatInputVector = new double[boardSize * boardSize * boardSize * boardSize * 4];
        int index = 0;
        for (int j = 0; j < boardSize * boardSize; j++) {
            for (int k = 0; k < boardSize; k++) {
                for (int l = 0; l < boardSize; l++) {
                    for (int m = 0; m < 4; m++) {
                        flatInputVector[index++] = tensor[j][k][l][m];
                    }
                }
            }
        }
        return Nd4j.create(flatInputVector, new int[]{1, boardSize * boardSize, boardSize, boardSize,4});
    }

    /**
     * returns the flat index given a row and col as specified below for a size of 3.
     * If the size is different it goes rows first the columns as shown
     * <p>
     *    0|1|2<br>
     *    3|4|5<br>
     *    6|7|8<br>
     * </p>
     * @param row the row
     * @param col the column
     * @param boardSize the size of the board
     * @return the flat index
     */
    public static int getBoardIndex(int row, int col, int boardSize) {
        return row * boardSize + col;
    }
}
