package com.hottes.caleb.ultimateticktacktoe.ui;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.Resources.PlayerType;
import com.hottes.caleb.ultimateticktacktoe.SubBoardState;
import com.hottes.caleb.ultimateticktacktoe.UltimateTickTackToe;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Pos;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

import java.util.Optional;

/**
 * Handles all the control flow for a two player game. Calls render one board state if nessacary. When game is over or user want to terminate it
 * will call appropriate methods to go back to menu or start a new two player game.
 */
public class GameController {

    private static Thread gameThread;
    private final ImageView gameView;
    private final ImageView playerOneImageView;
    private final ImageView playerTwoImageView;
    private final String playerOneName;
    private final String playerTwoName;
    private final PlayerType playerOneType;
    private final PlayerType playerTwoType;
    private BoardState boardState;
    private VBox theNode;
    private boolean isGameRunning = false;
    private Optional<EvaluatorConfiguration> playerOneEvaluatorConfig;
    private Optional<EvaluatorConfiguration> playerTwoEvaluatorConfig;
    private MCTSEvaluator evaluator;


    public GameController(String player1Name, String player2name, int boardSize, PlayerType playerOneType, PlayerType playerTwoType, Optional<EvaluatorConfiguration> playerOneEval, Optional<EvaluatorConfiguration> playerTWoEval) {
        //save constnats
        boardState = new BoardState(boardSize);
        //boardState = getTestState();
        this.playerOneName = player1Name;
        this.playerTwoName = player2name;
        this.theNode = new VBox();
        //initlaize the game view
        this.gameView = new ImageView();
        this.playerOneImageView = new ImageView();
        this.playerTwoImageView = new ImageView();
        this.playerOneType = playerOneType;
        this.playerTwoType = playerTwoType;


        //i know repeated code, can I just do it this once?
        if (this.playerOneType == PlayerType.COMPUTER) {
            playerOneEval.ifPresentOrElse(mctsEvaluator -> playerOneEvaluatorConfig = Optional.of(mctsEvaluator), () -> {
                throw new IllegalArgumentException("Player One was specified to be a computer but not evaluator configuration was provided.");
            });
        } else {
            playerOneEvaluatorConfig = Optional.empty();
        }

        if (this.playerTwoType == PlayerType.COMPUTER) {
            playerTWoEval.ifPresentOrElse(mctsEvaluator -> playerTwoEvaluatorConfig = Optional.of(mctsEvaluator), () -> {
                throw new IllegalArgumentException("Player Two was specified to be a computer but not evaluator configuration was provided.");
            });
        } else {
            playerTwoEvaluatorConfig = Optional.empty();
        }
        setUpGUI();
        render();

    }

    public static BoardState getTestState() {
        SubBoardState oneWin = new SubBoardState(new double[][]{
                {1, 1, 1},
                {0, 0, 0},
                {0, 0, 0}},
                false, 3);

        SubBoardState twoWin = new SubBoardState(new double[][]{
                {-1, -1, -1},
                {0, 0, 0},
                {0, 0, 0}},
                false, 3);

        SubBoardState empty = new SubBoardState(new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}},
                false, 3);

        SubBoardState inProgress = new SubBoardState(new double[][]{
                {1, -1, 1},
                {0, -1, 0},
                {1, 1, -1}},
                true, 3);

        SubBoardState draw = new SubBoardState(new double[][]{
                {1, -1, 1},
                {1, -1, -1},
                {-1, 1, 1}},
                false, 3);
        SubBoardState topRight = new SubBoardState(new double[][]{
                {-1, 0, -1},
                {-1, 0, -1},
                {0, 1, 1}},
                false, 3);
        SubBoardState activeBoard = new SubBoardState(new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {1, 1, 0}},
                false, 3);
        SubBoardState bottomLeft = new SubBoardState(new double[][]{
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 0}},
                false, 3);
        //have to create new boards so seperated boards in the board state are not tied to the same memory location.
        //if not then setting board 1,0 active will set all of the one win boards active because they all point to the same object.
        BoardState testState = new BoardState(new SubBoardState[][]{
                {new SubBoardState(oneWin.getStateAsCopy(), false, 3), new SubBoardState(oneWin.getStateAsCopy(), false, 3), topRight},
                {new SubBoardState(oneWin.getStateAsCopy(), false, 3), twoWin, activeBoard},
                {bottomLeft, new SubBoardState(empty.getStateAsCopy(), false, 3), new SubBoardState(empty.getStateAsCopy(), false, 3)}},
                3);
        testState.setAllBoardsActivity(false);
        testState.setBoardActive(1, 2);
        testState.setPlayerOneTurn(true);
        return testState;

    }

    private void setUpGUI() {
        this.getPane().getChildren().add(gameView);
        this.gameView.fitWidthProperty().bind(UltimateTickTackToe.widthProperty);
        this.gameView.fitHeightProperty().bind(UltimateTickTackToe.heightProperty.subtract(Resources.IN_GAME_INFO_BAR_HEIGHT));
        this.gameView.addEventHandler(MouseEvent.MOUSE_CLICKED, this::handleMouseEvent);

        //create the info bar at the bottom
        BorderPane infoPane = new BorderPane();
        playerOneImageView.setImage(SwingFXUtils.toFXImage(Resources.XImage, null));
        playerTwoImageView.setImage(SwingFXUtils.toFXImage(Resources.OImage, null));

        //set up player image on the left
        VBox leftBox = new VBox();
        Label playerOneLabel = new Label(playerOneName);
        playerOneImageView.fitHeightProperty().bind(playerOneLabel.heightProperty().multiply(-1).add(Resources.IN_GAME_INFO_BAR_HEIGHT));
        playerOneImageView.setPreserveRatio(true);
        leftBox.setAlignment(Pos.CENTER);
        leftBox.getChildren().addAll(playerOneImageView, playerOneLabel);
        infoPane.setLeft(leftBox);

        //set up player image on the right
        VBox rightBox = new VBox();
        Label playerTwoLabel = new Label(playerTwoName);
        playerTwoImageView.fitHeightProperty().bind(playerOneImageView.fitHeightProperty());
        playerTwoImageView.setPreserveRatio(true);
        rightBox.setAlignment(Pos.CENTER);
        rightBox.getChildren().addAll(playerTwoImageView, playerTwoLabel);
        infoPane.setRight(rightBox);


        Button newGameButton = new Button("New Game");
        newGameButton.setOnAction(_ -> {
            this.stopGame();//just to break ouf of the infinite and get things off the call stack.
            UltimateTickTackToe.startGame(playerOneName, playerOneName, playerOneType, playerOneEvaluatorConfig, playerTwoType, playerTwoEvaluatorConfig);
        });


        Button undoButton = new Button();
        undoButton.setDisable(true);
        undoButton.setOnAction(actionEvent -> System.out.println("Undoing move"));
        if (Resources.undoImage != null) {
            ImageView undoView = new ImageView(SwingFXUtils.toFXImage(Resources.undoImage, null));
            undoView.setFitHeight(Resources.IN_GAME_INFO_BAR_HEIGHT * .5);
            undoView.setPreserveRatio(true);
            undoButton.setGraphic(undoView);
        } else {
            //in case the program was unable to load the image
            undoButton.setText("Undo");
        }
        Button redoButton = new Button();
        redoButton.setDisable(false);
        redoButton.setOnAction(actionEvent -> redoAction());
        if (Resources.redoImage != null) {
            ImageView redoView = new ImageView(SwingFXUtils.toFXImage(Resources.redoImage, null));
            redoView.setFitHeight(Resources.IN_GAME_INFO_BAR_HEIGHT * .5);
            redoView.setPreserveRatio(true);
            redoButton.setGraphic(redoView);
        } else {
            //in case the program was unable to load the image
            redoButton.setText("Redo");
        }

        Button menuButton = new Button("Menu");
        menuButton.setOnAction(_ -> {
            this.stopGame();
            UltimateTickTackToe.backToMenuScreen();
        });
        HBox buttonBox = new HBox(undoButton, menuButton, newGameButton, redoButton);
        buttonBox.setAlignment(Pos.CENTER);
        buttonBox.setSpacing(10);
        infoPane.setCenter(buttonBox);
        this.getPane().getChildren().add(infoPane);
    }

    //game loop that runs while the game is running
    public void enterGameLoop() {
        this.isGameRunning = true;
        gameThread = new Thread(() -> {
            System.out.println("Game thread starting");
            while (isGameRunning) {
                //just to not overload the cpu
                //also nesscary for the thread to end properly for some reason.
                //this literally makes my amd ryzen 9 7000 at 5,2 GHz go from 11% utilization to 0 under idle. (for this process`)
                try {
                    Thread.sleep(1);
                } catch (InterruptedException _) {

                }

                if ((boardState.isPlayerOneTurn() && playerOneType == PlayerType.HUMAN) || (!boardState.isPlayerOneTurn() && playerTwoType == PlayerType.HUMAN)) {
                    //if we are waiting on a human to play then do nothing.
                    continue;
                } else {
                    //so now the computer has to play.
                    //player one, X is always denoted by X here. If it is currently player one's turn then we can just give the evaluator the board as is, however if it is player two's turn
                    //then the board needs to have the marker 1 show the player whose turn it is to move, so we need to invert the board.
                    boolean inversioNeeded = !boardState.isPlayerOneTurn();
                    BoardState stateToPass = boardState.getClone();
                    if (inversioNeeded) {
                        stateToPass.invertState();
                    }
                    evaluator = new MCTSEvaluator(stateToPass, boardState.isPlayerOneTurn() ? playerOneEvaluatorConfig.get() : playerTwoEvaluatorConfig.get());
                    if (!processPlayerInput((UltimateTickTacToeGameAction) evaluator.preformSearch())) {
                        boardState.togglePlayerOneTurn();
                        Platform.runLater(() -> {
                            Alert alert = new Alert(Alert.AlertType.INFORMATION, "Computer attempted to make illegal action, their turn is forfeit.");
                            alert.show();//it might be playing autonomously with the user off doing something else.

                        });
                    }
                }


            }
            System.out.println("Game Thread ending");
        });
        gameThread.start();


    }

    public void stopGame() {
        if (evaluator != null) {
            evaluator.forcePlay();
        }
        this.isGameRunning = false;
    }

    private void handleMouseEvent(MouseEvent event) {
        //only process clicks to the board if the game is actually being played.
        if (!isGameRunning) {
            return;
        }
        if ((boardState.isPlayerOneTurn() && playerOneType == PlayerType.COMPUTER) || (!boardState.isPlayerOneTurn() && playerTwoType == PlayerType.COMPUTER)) {
            //if it is the computers turn than don't process clicks on the board.
            return;
        }
        for (int i = 0; i < boardState.getSubBoardBoundingBoxes().length; i++) {
            for (int j = 0; j < boardState.getSubBoardBoundingBoxes()[0].length; j++) {
                if ((boardState.getSubBoardBoundingBoxes()[i][j] != null) &&
                        (boardState.getSubBoardBoundingBoxes()[i][j].contains(event.getSceneX(), event.getSceneY()))) {
                    //then we are on the right sub board

                    SubBoardState subBoard = boardState.getMinorBoards()[i][j];
                    for (int k = 0; k < subBoard.getRows(); k++) {
                        for (int l = 0; l < subBoard.getCols(); l++) {
                            if ((subBoard.subBoardBoundingBoxes[k][l] != null) &&
                                    (subBoard.subBoardBoundingBoxes[k][l].contains(event.getSceneX(), event.getSceneY()))) {
                                //then we have found the square the user clicked on
                                System.out.println("Input detected: SubBoard (" + i + ", " + j + ") Square(" + l + ", " + k + ")");
                                processPlayerInput(new UltimateTickTacToeGameAction(i, j, l, k, boardState.isPlayerOneTurn() ? 1 : -1));


                            }
                        }
                    }

                }
            }
        }
    }

    /**
     * Assuming that the player who proposed this action is alloed to do so, determines if the move is legal and if so plays it
     * The method calls render to make sure that the display is up to date.
     * This method will also exit the game loop if this move ends the game.
     *
     * @param proposedAction the action that a player would like to take
     * @return if the move was played or not.
     */
    private boolean processPlayerInput(UltimateTickTacToeGameAction proposedAction) {
        proposedAction.setMarker(boardState.isPlayerOneTurn() ? 1 : -1);//the evaluator always sees things as if its player one, so compensate for that.
        System.out.print("New game Action Proposed: ");
        System.out.print(proposedAction);
        System.out.println(" with marker: " + proposedAction.getMarker());
        boolean movePlayed = false;
        //now we need to validate and see if this action is legal
        if (boardState.isMoveLegal(proposedAction)) {
            //preform the action
            boardState.preformAction(proposedAction);
            movePlayed = true;
            //because we took an action we also need to see if the game has ended.
            double eval = boardState.getEvaluation();
            if (eval != BoardState.Evaluation.IN_PROGRESS.getlabel()) {
                System.out.println("Game is over input no longer accepted");
                this.stopGame();
                Platform.runLater(() -> {
                    Alert alert;
                    if (eval == BoardState.Evaluation.DRAW.getlabel()) {
                        alert = new Alert(Alert.AlertType.INFORMATION, "The game ended in a draw!");
                    } else {
                        alert = new Alert(Alert.AlertType.INFORMATION, ((eval == 1) ? playerOneName : playerTwoName) + " Won!");
                    }
                    alert.showAndWait();
                });

            }
        }
        render();
        return movePlayed;
    }

    public void render() {
        //set the image views to the new redered images
        gameView.setImage(SwingFXUtils.toFXImage(boardState.getRenderedImage(gameView.getFitWidth(), gameView.getFitHeight()), null));
        if (boardState.isPlayerOneTurn()) {
            playerOneImageView.setImage(SwingFXUtils.toFXImage(Resources.XSelectedImage, null));
            playerTwoImageView.setImage(SwingFXUtils.toFXImage(Resources.OImage, null));
        } else {
            playerOneImageView.setImage(SwingFXUtils.toFXImage(Resources.XImage, null));
            playerTwoImageView.setImage(SwingFXUtils.toFXImage(Resources.OSelectedImage, null));
        }

    }

    public Pane getPane() {
        return theNode;
    }

    public boolean isGameRunning() {
        return isGameRunning;
    }

    public void redoAction() {

    }

}
