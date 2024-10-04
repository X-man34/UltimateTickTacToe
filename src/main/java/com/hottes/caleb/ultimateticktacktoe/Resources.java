package com.hottes.caleb.ultimateticktacktoe;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.lang.instrument.Instrumentation;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;

public class Resources {



    //constants to define how much spacing is put between boards based on the window dimensions
    public static final double HORIZONTAL_SPACING_FACTOR = .01;
    public static final double VERTICAL_SPACING_FACTOR = HORIZONTAL_SPACING_FACTOR;
    //what percent of the height is devoted to the information bar at the bottom of the game.
    public static final int IN_GAME_INFO_BAR_HEIGHT = 120;
    public static final double PLAYER_IMAGE_WIDTH_FACTOR = .15;
    public static final Paint ACTIVE_BOARD_COLOR = new Color(0, 1, 255, 133);
    public static final int LINE_STROKE = 4;
    public static final int BOARD_SIZE = 3;
    public static final BufferedImage playerImage;
    public static final BufferedImage undoImage;
    public static final BufferedImage redoImage;
    public static final BufferedImage XImage;
    public static final BufferedImage OImage;
    public static final BufferedImage XSelectedImage;
    public static final BufferedImage OSelectedImage;
    public static final int EVALUATION_ITERATION_HARD_LIMIT = 50000000;
    public static final int MAX_ITERS_USER_CAN_ENTER = 10000000;

    public static final EvaluatorConfiguration DEFAULT_EVALUATOR_CONFIGURATION = new EvaluatorConfiguration(2, 1000, 10, 5, 0, true, false);
    public static final EvaluatorConfiguration EASY_EVALUATOR_CONFIGURATION = new EvaluatorConfiguration(2, 1000, 10, 1, 100, true, false);
    public static final EvaluatorConfiguration MEDIUM_EVALUATOR_CONFIGURATION = new EvaluatorConfiguration(2, 1000, 30, 5, 50, true, false);
    public static final EvaluatorConfiguration HARD_EVALUATOR_CONFIGURATION = new EvaluatorConfiguration(2, 1000, 60, 25, 25, false, true);
    public static final int MULTTHREADED_BATCH_SIZE = 100000;
    public enum PlayerType {
        HUMAN,
        COMPUTER
    }

    public enum Evaluation {
        IN_PROGRESS(0),
        DRAW(-.25),
        PLAYER_ONE_WIN(1),
        PLAYER_TWO_WIN(-1);
        private final double label;
        Evaluation(double val) {
            label = val;
        }

        public double getlabel() {
            return label;
        }
    }



    static {
        BufferedImage tempVar;
        BufferedImage tempRedo;
        BufferedImage tempUndo;
        BufferedImage tempX;
        BufferedImage tempO;
        BufferedImage tempXSelected;
        BufferedImage tempOSelected;
        //get the image for the player icons

        try {
            tempVar = ImageIO.read(Objects.requireNonNull(Resources.class.getResourceAsStream("playerIcon.png")));
            tempUndo = ImageIO.read(Objects.requireNonNull(GameController.class.getResourceAsStream("undo.png")));
            tempRedo = ImageIO.read(Objects.requireNonNull(GameController.class.getResourceAsStream("redo.png")));
            tempX = ImageIO.read(Objects.requireNonNull(GameController.class.getResourceAsStream("X.png")));
            tempO = ImageIO.read(Objects.requireNonNull(GameController.class.getResourceAsStream("O.png")));
            tempXSelected = ImageIO.read(Objects.requireNonNull(GameController.class.getResourceAsStream("XSelected.png")));
            tempOSelected = ImageIO.read(Objects.requireNonNull(GameController.class.getResourceAsStream("OSelected.png")));

        } catch (
                IOException e) {
            e.printStackTrace();
            tempVar = new BufferedImage(100,100, BufferedImage.TYPE_INT_ARGB);//size doesn't matter it gets scaled later anyways.
            tempUndo = null;
            tempRedo = null;
            tempX = null;
            tempO = null;
            tempXSelected = null;
            tempOSelected = null;
        }
        playerImage = tempVar;
        undoImage = tempUndo;
        redoImage = tempRedo;
        XImage = tempX;
        OImage = tempO;
        XSelectedImage = tempXSelected;
        OSelectedImage = tempOSelected;
    }


    /**
     * Changes all pixels of an old color into a new color, preserving the
     * alpha channel.
     * @see <a href="https://codereview.stackexchange.com/questions/146609/color-substitution-in-a-bufferedimage">Source</a>
     */
    public static BufferedImage changeColorBUfferedImage(
            BufferedImage imgBuf,
            int oldRed, int oldGreen, int oldBlue,
            int newRed, int newGreen, int newBlue) {

        int RGB_MASK = 0x00ffffff;
        int ALPHA_MASK = 0xff000000;

        int oldRGB = oldRed << 16 | oldGreen << 8 | oldBlue;
        int toggleRGB = oldRGB ^ (newRed << 16 | newGreen << 8 | newBlue);

        int w = imgBuf.getWidth();
        int h = imgBuf.getHeight();

        int[] rgb = imgBuf.getRGB(0, 0, w, h, null, 0, w);
        for (int i = 0; i < rgb.length; i++) {
            if ((rgb[i] & RGB_MASK) == oldRGB) {
                rgb[i] ^= toggleRGB;
            }
        }
        imgBuf.setRGB(0, 0, w, h, rgb, 0, w);
        return deepCopy(imgBuf);
    }

    static BufferedImage deepCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(null);
        return new BufferedImage(cm, raster, isAlphaPremultiplied, null);
    }

    public static BufferedImage scaleImage(int newWidth, int newHeight, BufferedImage original) {
        BufferedImage resized = new BufferedImage(newWidth, newHeight, original.getType());
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, newWidth, newHeight, 0, 0, original.getWidth(),
                original.getHeight(), null);
        g.dispose();
        return resized;
    }

    /**
     * Draws a red X if the token is 1 and a green O it the token is -1.
     * For other token values it returns false.
     *
     * @param g the graphics context to draw with
     * @param boundingBox the bouding box we are allowed to draw in
     * @param token the token we are potentially drawing
     * @return whether or not something was drawn
     */
    public static boolean drawToken(Graphics2D g, Rectangle2D.Double boundingBox, double token) {
        if (token == 0) {
            return false;
        }
        g.setStroke(new BasicStroke(8));
        //g.drawRect((int) Math.round(boundingBox.x), (int) Math.round(boundingBox.y),(int) Math.round(boundingBox.width),(int)  Math.round(boundingBox.height));
        double shrinkfactor = .85;//have to shrink the bounding boxes a bit so they don't look bad
        double centerX = boundingBox.getCenterX();
        double centerY = boundingBox.getCenterY();

        double newWidth = shrinkfactor * boundingBox.width;
        double newHeight = shrinkfactor * boundingBox.height;
        Rectangle2D.Double scaledBox = new Rectangle2D.Double(centerX - (newWidth / 2), centerY - (newHeight / 2), newWidth, newHeight);
        if (token == 1) {
            g.setPaint(Color.RED);
            g.drawLine((int) Math.round(scaledBox.x), (int) Math.round(scaledBox.y), (int) Math.round(scaledBox.x + scaledBox.width), (int) Math.round(scaledBox.y + scaledBox.height));
            g.drawLine((int) Math.round(scaledBox.x + scaledBox.width), (int) Math.round(scaledBox.y), (int) Math.round(scaledBox.x), (int) Math.round(scaledBox.y + scaledBox.height));
            return true;
        }else if (token == -1){
            g.setPaint(Color.GREEN);
            g.drawOval((int) Math.round(scaledBox.x), (int) Math.round(scaledBox.y),(int) Math.round(scaledBox.width),(int)  Math.round(scaledBox.height));
            return true;
        }else {
            return false;
        }
    }
    /**
     *
     * Evaluations are 1 for player 1 wins, -1 for player 1 loses, 0 for intederminate state and -.25 for draw
     *Copy past from copilot LLM
     * this method is called roughly 350 times on average per iteration so we're looking at ~400 million calls for 1.2million iterations.
     * @return the static evaluation of the game state
     */
    public static double getTicTacToeEvaluationBruteForce(double[][] state) {
        // Check rows, columns, and diagonals for a win
        for (int i = 0; i < 3; i++) {
            if (state[i][0] != 0 && state[i][0] == state[i][1] && state[i][1] == state[i][2]) {
                return (state[i][0] == 1) ? Evaluation.PLAYER_ONE_WIN.label : Evaluation.PLAYER_TWO_WIN.label;
            }
            if (state[0][i] != 0 && state[0][i] == state[1][i] && state[1][i] == state[2][i]) {
                return (state[0][i] == 1) ? Evaluation.PLAYER_ONE_WIN.label : Evaluation.PLAYER_TWO_WIN.label;
            }
        }

        // Check diagonals
        if (state[0][0] != 0 && state[0][0] == state[1][1] && state[1][1] == state[2][2]) {
            return (state[0][0] == 1) ? Evaluation.PLAYER_ONE_WIN.label : Evaluation.PLAYER_TWO_WIN.label;
        }
        if (state[0][2] != 0 && state[0][2] == state[1][1] && state[1][1] == state[2][0]) {
            return (state[0][2] == 1) ? Evaluation.PLAYER_ONE_WIN.label : Evaluation.PLAYER_TWO_WIN.label;
        }

        // Check if the state is full (draw) or still in progress
        boolean isFull = true;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] == 0) {
                    isFull = false;
                    break;
                }
            }
        }
        return isFull ? Evaluation.DRAW.label: Evaluation.IN_PROGRESS.label;
    }

    /**
     * creates a hashcode for a tick tac toe board. Assumes that each element of the array is either a 1 for player 1, -1 for player two or anything else for nothing.
     * this is to be used for fast evaluation functions using lookuptables
     * @param state the state variable from a game state object
     * @return the hashcode for static evaluation for this board.
     */
    public static int fastTicTacToeHashcode(double[][] state) {
        //constructs a string representation of the state using base 3 and converts it to an integer
        StringBuilder ternaryRepresentation = new StringBuilder();
        for (double[] doubles : state) {
            for (int j = 0; j < doubles.length; j++) {
                ternaryRepresentation.append(((int) Math.round(doubles[j])) + 2);
            }
        }
        return Integer.parseInt(ternaryRepresentation.toString(), 4);
    }

    public static void generateMatrixPermutations(double[][] matrix, int row, int col, List<double[][]> result) {
        if (row == 3) {
            // Matrix is complete; add it to the result
            double[][] copy = new double[3][3];
            for (int i = 0; i < 3; i++) {
                System.arraycopy(matrix[i], 0, copy[i], 0, 3);
            }
            result.add(copy);
            return;
        }

        for (int val = -1; val <= 1; val++) {
            matrix[row][col] = val;
            int nextRow = row + (col + 1) / 3;
            int nextCol = (col + 1) % 3;
            generateMatrixPermutations(matrix, nextRow, nextCol, result);
        }
    }


    public static String getBitSetAsString(BitSet bi) {
        StringBuilder s = new StringBuilder();
        for( int i = 0; i < bi.length();  i++ )
        {
            s.append( bi.get(i) ? 1: 0 );
        }

       return s.toString();
    }


}
