package com.hottes.caleb.ultimateticktacktoe.ui;

import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import javafx.collections.FXCollections;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;

import java.util.Optional;


/**
 * UI class for the settings for a player
 * If the user chooses to run this player as a human than its pretty straitforeward.
 * if its a bot though then all the settings are tunable here. There are some preset options but the goal of this class is to allow the user to instantiate a bot however they want.
 */
public class PlayerBox extends VBox {

    private final TextField nameField;
    private final ComboBox<Resources.PlayerType> playerTypeComboBox;
    private final ComboBox<Difficulty> difficultyComboBox;
    private final Spinner<Integer> timeSpinner;
    private final Spinner<Integer> iterationsSpinner;
    private final Spinner<Integer> threadsSpinner;
    private final Spinner<Double> cSpinner;
    private final Slider stupidSlider;
    private final CheckBox allowForcePlay;
    private final CheckBox maxMyCPU;
    private final RadioButton timeButton;

    public PlayerBox(String playerNum) {


        double spacing = 10;
        this.setSpacing(spacing);
        this.setAlignment(Pos.CENTER);
        this.getChildren().add(new Label("         Player " + playerNum));

        nameField = new TextField("Player " + playerNum);
        nameField.setEditable(true);
        nameField.setPrefColumnCount(10);
        HBox nameBox = new HBox(new Label("Name: "), nameField);
        nameBox.setSpacing(spacing);
        nameBox.setAlignment(Pos.CENTER);
        this.getChildren().add(nameBox);

        playerTypeComboBox = new ComboBox<>(FXCollections.observableArrayList(Resources.PlayerType.HUMAN, Resources.PlayerType.COMPUTER));
        HBox typeBox = new HBox(new Label("Type: "), playerTypeComboBox);
        typeBox.setSpacing(spacing);
        typeBox.setAlignment(Pos.CENTER);

        this.getChildren().add(typeBox);

        difficultyComboBox = new ComboBox<>(FXCollections.observableArrayList(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD));
        difficultyComboBox.getSelectionModel().clearAndSelect(0);
        HBox difficultyBox = new HBox(new Label("Difficulty: "), difficultyComboBox);
        difficultyBox.setSpacing(spacing);
        difficultyBox.setAlignment(Pos.CENTER);
        this.getChildren().add(difficultyBox);


        timeButton = new RadioButton("Time: ");
        timeSpinner = new Spinner<>(0, 120, 30);
        timeSpinner.setEditable(true);
        RadioButton iterationsButton = new RadioButton("Iterations: ");
        iterationsSpinner = new Spinner<>(1000, Resources.MAX_ITERS_USER_CAN_ENTER, 100000);
        iterationsSpinner.setEditable(true);


        ToggleGroup toggleGroup = new ToggleGroup();
        toggleGroup.getToggles().addAll(timeButton, iterationsButton);
        toggleGroup.selectToggle(timeButton);

        HBox timeHBox = new HBox(timeButton, timeSpinner);
        timeHBox.setAlignment(Pos.BASELINE_RIGHT);
        maxMyCPU = new CheckBox("Max out my cpu!");

        VBox endConditionsListBox = new VBox(timeHBox, maxMyCPU);
        endConditionsListBox.setAlignment(Pos.CENTER);
        endConditionsListBox.setSpacing(spacing);
        HBox endConditionBox = new HBox(new Label("End Condition:   "), endConditionsListBox);
        endConditionBox.setAlignment(Pos.CENTER);

        threadsSpinner = new Spinner<>(1, 100, 5);
        threadsSpinner.disableProperty().bind(maxMyCPU.selectedProperty());
        threadsSpinner.setEditable(true);
        cSpinner = new Spinner<>(0.00000001, 5, 2);
        stupidSlider = new Slider(0, 100, 0);
        stupidSlider.setBlockIncrement(1);

        allowForcePlay = new CheckBox("Allow Force Play: ");
        allowForcePlay.setSelected(true);


        HBox threadsBox = new HBox(new Label("Threads: "), threadsSpinner);
        threadsBox.setAlignment(Pos.CENTER);
        HBox cHBox = new HBox(new Label("C Value: "), cSpinner);
        cHBox.setAlignment(Pos.CENTER);
        HBox stupidHBox = new HBox(new Label("Stupidity: "), stupidSlider);
        stupidHBox.setAlignment(Pos.CENTER);
        VBox customSettingsBox = new VBox(endConditionBox, threadsBox, stupidHBox, allowForcePlay);
        customSettingsBox.setSpacing(spacing);
        customSettingsBox.setDisable(true);

        customSettingsBox.setAlignment(Pos.CENTER);

        CheckBox customCheckbox = new CheckBox("Custom: ");
        customSettingsBox.visibleProperty().bind(customCheckbox.selectedProperty());
        customCheckbox.setSelected(false);

        customCheckbox.setOnAction(_ -> {
            customSettingsBox.setDisable(!customCheckbox.isSelected());
            difficultyBox.setDisable(customCheckbox.isSelected());
        });
        playerTypeComboBox.getSelectionModel().selectedItemProperty().addListener((_, _, _) -> customCheckbox.setDisable(playerTypeComboBox.getValue() == Resources.PlayerType.HUMAN));
        playerTypeComboBox.getSelectionModel().clearAndSelect(0);
        this.getChildren().add(customCheckbox);
        this.getChildren().add(new Separator(Orientation.HORIZONTAL));
        this.getChildren().add(customSettingsBox);

    }

    /**
     * for reading the settings and getting the approroatly configured evaluator.
     *
     * @return could be null if this PlayerBox is configured to human mode. check before calling.
     */
    public EvaluatorConfiguration getEvalulatorConfig() {
        if (playerTypeComboBox.getValue() == Resources.PlayerType.HUMAN) {
            return null;
        } else if (difficultyComboBox.isDisabled()) {
            //then this is a custom game
            return new EvaluatorConfiguration(cSpinner.getValue(), 1000, timeSpinner.getValue(), threadsSpinner.getValue(), (int) Math.round(stupidSlider.getValue()), allowForcePlay.isSelected(), maxMyCPU.isSelected(), Optional.empty());//assumes a board size of three for now
        } else {
            switch (difficultyComboBox.getValue()) {
                case MEDIUM -> {
                    return Resources.MEDIUM_EVALUATOR_CONFIGURATION;
                }
                case HARD -> {
                    return Resources.HARD_EVALUATOR_CONFIGURATION;
                }
                case EASY -> {
                    return Resources.EASY_EVALUATOR_CONFIGURATION;
                }
                default -> {//easy
                    //defaults to defaults lol
                    return Resources.DEFAULT_EVALUATOR_CONFIGURATION;
                }
            }
        }

    }

    public String getPlayerName() {
        return nameField.getText();
    }

    public Resources.PlayerType getPlayerType() {
        return playerTypeComboBox.getValue();
    }

    private enum Difficulty {
        EASY,
        MEDIUM,
        HARD
    }

}
