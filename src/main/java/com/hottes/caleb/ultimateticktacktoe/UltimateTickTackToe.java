package com.hottes.caleb.ultimateticktacktoe;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import com.hottes.caleb.ultimateticktacktoe.ui.MenuScreen;
import javafx.application.Application;
import javafx.beans.property.ReadOnlyDoubleProperty;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.Optional;


/**
 * Things to add
 * 1. Play timers
 * 2. tutorial
 * 3. move ratings
 * 4. currently useing brute force to evaluate boards, quick hashing is easily collided, and sha is too slow. Use ternary converted to base ten to have determistic and fast
 * hash function.
 *5. undo/redo buttons
 * 6. load menu screen options from config file
 * 7. more documentation
 *
 */


public class UltimateTickTackToe extends Application {

    public static BoardState theMainBoard;

    private static  MenuScreen menu;
    private static Scene scene;
    private static Stage theStage;
    private static GameController currentGameController;
    public static ReadOnlyDoubleProperty widthProperty;
    public static ReadOnlyDoubleProperty heightProperty;

    public enum Screens {
        MENU,
        SINGLEPLAYER,
        TWO_PLAYER
    }

    @Override
    public void start(Stage stage) throws IOException {
        theStage = stage;
        theMainBoard = new BoardState(Resources.BOARD_SIZE);
        menu = new MenuScreen();
        scene = new Scene(menu);


        widthProperty = scene.widthProperty();
        heightProperty = scene.heightProperty();
        theStage.setScene(scene);
        theStage.show();
        theStage.setMinHeight(Resources.IN_GAME_INFO_BAR_HEIGHT + 1);

        scene.widthProperty().addListener((observableValue, number, t1) -> {
            if (currentGameController != null) {
                currentGameController.render();
            }
        });

        scene.heightProperty().addListener((_, number, _) -> {
            if (currentGameController != null) {
                currentGameController.render();
            }
        });


    }

    public static void startGame(String playerOneName, String playerTwoName, Resources.PlayerType playerOneType, Optional<EvaluatorConfiguration> playerOneEvaluator, Resources.PlayerType playerTwoType, Optional<EvaluatorConfiguration> playerTwoEvaluator) {
        currentGameController = new GameController(playerOneName, playerTwoName, Resources.BOARD_SIZE, playerOneType, playerTwoType, playerOneEvaluator, playerTwoEvaluator);
        scene.setRoot(currentGameController.getPane());
        theStage.setTitle("Ultimate Tick Tac Toe: " + playerOneName + " vs. " + playerTwoName);
        currentGameController.enterGameLoop();
    }

    public static void backToMenuScreen() {
        scene.setRoot(menu);
        theStage.setTitle("Menu");
        currentGameController.stopGame();
    }




    public static void main(String[] args) {
        launch();
    }
}