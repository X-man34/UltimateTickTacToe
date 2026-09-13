package com.hottes.caleb.ultimateticktacktoe.ui;

import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.UltimateTickTackToe;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;

import java.util.Optional;

public class MenuScreen extends VBox {

    public MenuScreen() {
        this.setPrefHeight(1000);
        this.setPrefWidth(1000);
        this.setAlignment(Pos.CENTER);

        Label mainLabel = new Label("Ultimate Tick Tac Toe");
        mainLabel.setFont(new Font(72));

        Button startGameButton = new Button("Start Game");
        startGameButton.setFont(new Font(48));


        PlayerBox playerOneBox = new PlayerBox("One");
        PlayerBox playerTwoBox = new PlayerBox("Two");
        HBox settingsBox = new HBox(playerOneBox, new Separator(Orientation.VERTICAL), playerTwoBox);
        settingsBox.setSpacing(20);
        settingsBox.setAlignment(Pos.CENTER);

        //lets go!!! giant unreadable one liners!!
        startGameButton.setOnAction(_ -> UltimateTickTackToe.startGame(playerOneBox.getPlayerName(), playerTwoBox.getPlayerName(), playerOneBox.getPlayerType(), playerOneBox.getPlayerType() == Resources.PlayerType.HUMAN ? Optional.empty() : Optional.of(playerOneBox.getEvalulatorConfig()), playerTwoBox.getPlayerType(), playerTwoBox.getPlayerType() == Resources.PlayerType.HUMAN ? Optional.empty() : Optional.of(playerTwoBox.getEvalulatorConfig())));
        this.setSpacing(20);
        this.getChildren().addAll(mainLabel, startGameButton, settingsBox);
    }

}
