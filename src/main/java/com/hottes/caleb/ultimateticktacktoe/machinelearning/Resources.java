package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import org.apache.commons.io.filefilter.DelegateFileFilter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Resources {



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
    public static INDArray getValueNetworkInputV1(BoardState boardState) {
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

    /**
     * takes in a board state and returns an IND array of shape [width, height, 3] which is basically a really low res image of the board.
     * player one will be red, player two is green and the background of active boards is white.
     * @param state
     * @return
     */
    public static BufferedImage getImageForAI(BoardState state, int scaleFactor) {
        int markerSize = 4 * scaleFactor;
        int mainBorderWith = (int) (3.0 / 4.0 * markerSize);
        int subBoardLineWith = (int) (.25 * markerSize);
        int majorBoardLineWidth = (int) (.5 * markerSize);
        int subBoardSize = 3 * markerSize + 10 * subBoardLineWith;
        int size = 3 * subBoardSize + 2 * mainBorderWith + 2 * majorBoardLineWidth;
        BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = image.createGraphics();
        g.setPaint(Color.WHITE);
        g.fillRect(0, 0, size, size);
        g.setPaint(Color.BLACK);
        //render the outside borders
        g.fillRect(0, 0, mainBorderWith, size);
        g.fillRect(0, 0, size, mainBorderWith);
        g.fillRect(0, size - mainBorderWith, size, mainBorderWith);
        g.fillRect(size - mainBorderWith, 0, mainBorderWith, size);
        for (int row = 0; row < 3; row++) {
            //render the major board borders
            g.setPaint(Color.BLACK);

            g.fillRect(0, mainBorderWith + (row + 1) * subBoardSize + row * majorBoardLineWidth, size, majorBoardLineWidth);
            g.fillRect(mainBorderWith + (row + 1) * subBoardSize + row * majorBoardLineWidth, 0, majorBoardLineWidth, size);
            for (int col = 0; col < 3; col++) {
                int columnOffset = mainBorderWith + (col) * subBoardSize + (col) * majorBoardLineWidth;
                int rowOffset = mainBorderWith + (row) * subBoardSize + row * majorBoardLineWidth;
                //for each sub board

                //check it its won or not and if so draw appropriate color over
                double eval = state.getMinorBoardAt(row, col).getEvaluation();
                if (eval != BoardState.Evaluation.IN_PROGRESS.getlabel()) {
                    //draw in terminal states
                    if (eval == BoardState.Evaluation.PLAYER_ONE_WIN.label) {
                        g.setPaint(Color.RED);
                    }else if (eval == BoardState.Evaluation.PLAYER_TWO_WIN.label) {
                        g.setPaint(Color.GREEN);

                    }else {
                        g.setPaint(Color.YELLOW);
                    }
                    g.fillRect(columnOffset + subBoardLineWith, rowOffset + subBoardLineWith, subBoardSize - subBoardLineWith - subBoardLineWith, subBoardSize - subBoardLineWith - subBoardLineWith);
                } else  {
                    if (state.getMinorBoardAt(row, col).isActive()) {
                        //if active draw in if its active
                        g.setPaint(Color.BLUE);
                        g.fillRect(columnOffset, rowOffset, subBoardSize, subBoardSize);
                    }

                    for (int minRow = 0; minRow < 3; minRow++) {

                        g.setPaint(Color.BLACK);
                        if (minRow < 2) {
                            g.fillRect(columnOffset + subBoardLineWith, rowOffset + subBoardLineWith + (minRow + 1) * (2 * subBoardLineWith + markerSize) + minRow * subBoardLineWith, subBoardSize - subBoardLineWith - subBoardLineWith, subBoardLineWith);
                            g.fillRect(columnOffset + (minRow + 1) * (markerSize + 3 * subBoardLineWith), rowOffset + subBoardLineWith, subBoardLineWith, subBoardSize - subBoardLineWith - subBoardLineWith);


                        }
                        for (int minCol = 0; minCol < 3; minCol++) {
                            int minColumnOffset = columnOffset + 2 * subBoardLineWith + (minCol) * markerSize + (minCol) * 3 * subBoardLineWith;
                            int minRowOffset = rowOffset + 2 * subBoardLineWith + (minRow) * markerSize + (minRow) * 3 * subBoardLineWith;
                            double val = state.getMinorBoardAt(row, col).itemAt(minRow, minCol);
                            if (val == 1) {
                                g.setPaint(Color.RED);
                                g.fillRect(minColumnOffset, minRowOffset, markerSize, markerSize);
                            } else if (val == -1) {
                                g.setPaint(Color.GREEN);
                                g.fillRect(minColumnOffset, minRowOffset, markerSize, markerSize);
                            }
                        }
                    }
                }
            }
        }
//        // Convert the BufferedImage to INDArray
//        int width = image.getWidth();
//        int height = image.getHeight();
//        int[] pixels = new int[width * height];
//        pixels = image.getRGB(0, 0, width, height, pixels, 0, width);
//
//        // Create an INDArray to store the pixel values
//        // We can create a 3D INDArray to store the RGB values for each pixel.
//        // Shape: [height, width, 3] (height x width x 3 for RGB channels)
//        INDArray indArray = Nd4j.create(height, width, 3);
//
//        for (int y = 0; y < height; y++) {
//            for (int x = 0; x < width; x++) {
//                int pixel = pixels[y * width + x];
//                // Extract RGB components
//                int red = (pixel >> 16) & 0xFF;
//                int green = (pixel >> 8) & 0xFF;
//                int blue = pixel & 0xFF;
//
//                // Set values in the INDArray
//                indArray.putScalar(new int[]{y, x, 0}, red);
//                indArray.putScalar(new int[]{y, x, 1}, green);
//                indArray.putScalar(new int[]{y, x, 2}, blue);
//            }
//        }


        return image;
    }

    private static int getIntRRB(int red,int green, int blue) {
        return  (red << 16) | (green << 8) | blue;
    }

    public static INDArray getPolicyNetworkInputV1(BoardState state) {
        return getValueNetworkInputV1(state);
    }

    public static File getDataFolderFromUser(Scanner scanner) {
        System.out.print("Enter filepath of data folder: ");
        String baseFolderInput = scanner.nextLine();

        File dataFolder = new File(baseFolderInput);
        if (!dataFolder.isDirectory() && dataFolder.exists()) {
            System.out.println("Invalid data folder path. Exiting. ");
            System.exit(1);
        }
        return dataFolder;
    }

    /**
     * takes in a flat vector from the policy network which represents the value assinged to all the moves on the board, ie, the raw output from the network.
     * This function uses the given board state to determine which moves are legal and returns the legal move with the highest value assigned to it by the neural network.
     *
     * @param output the neural network output
     * @param boardSize the size of the board
     * @param currentState the state of the board that was fed into the neural network
     * @return the game action that the network thinks is best.
     * @throws IllegalArgumentException if unable to find a legal move.
     */
    public static UltimateTickTacToeGameAction getBestActionFromPolicyNetworkOutput(double[] output, int boardSize, BoardState currentState) {
        double bestProb = Double.NEGATIVE_INFINITY;
        UltimateTickTacToeGameAction bestAction = null;
        for (int majorRow = 0; majorRow < boardSize; majorRow++) {
            for (int minorRow = 0; minorRow < boardSize; minorRow++) {
                for (int majorCol = 0; majorCol < boardSize; majorCol++) {
                    for (int minorCol = 0; minorCol < boardSize; minorCol++) {
                        UltimateTickTacToeGameAction actionHere = new UltimateTickTacToeGameAction(majorRow, majorCol, minorRow, minorCol, currentState.isPlayerOneTurn()?1:-1);
                        if (currentState.isLegal(actionHere)) {
                            double prob = output[getIndex(majorRow, majorCol, minorRow, minorCol, boardSize)];
                            if (prob > bestProb) {
                                bestProb = prob;
                                bestAction = actionHere;
                            }
                        }


                    }

                }
            }
        }
        if (bestAction == null) {
            throw new IllegalArgumentException("Could not find any legal actions");
        }
        return bestAction;
    }


    public static void timeComplexityTester(Runnable task,  int maxN, int threads, String name) {
        ArrayList<Long> times = new ArrayList<>();
        ArrayList<Integer> xVals = new ArrayList<>();
        ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads);
        for (int n = 0; n < maxN; n++) {
            threadPoolExecutor.submit(new NIterTester(times, n, task));
            xVals.add(n);
        }
        System.out.println("computing");
        while (times.size() < maxN) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {

            }
            System.out.println(times.size());
        }
        System.out.println("plotting");
        Plot plot = Plot.create();
        plot.plot().add(xVals, times);
        plot.title(name);
        plot.xlabel("Iterations");
        plot.ylabel("Time (ms)");
        try {
            plot.show();
        } catch (IOException | PythonExecutionException e) {
            e.printStackTrace();
        }
        threadPoolExecutor.shutdownNow();

    }


    private static class NIterTester implements Runnable {

        private final ArrayList<Long> times;
        private final long iters;
        private final Runnable task;
        public NIterTester(ArrayList<Long> theTimes, long numIters, Runnable theTask) {
            this.times = theTimes;
            this.iters = numIters;
            this.task = theTask;
        }

        @Override
        public void run() {
            long startTime = System.currentTimeMillis();
            for (int iteration = 0; iteration < iters; iteration++) {
                task.run();
            }
            long time = System.currentTimeMillis() - startTime;
            times.add(time);
        }
    }
}
