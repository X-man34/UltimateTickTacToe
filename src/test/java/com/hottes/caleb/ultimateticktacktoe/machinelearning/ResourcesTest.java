package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import org.apache.commons.lang3.mutable.Mutable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class ResourcesTest {

    @Test
    @Disabled
    void getValueNetworkInputV1_0() {

        BoardState testState = GameController.getTestState();
        //System.out.println(Resources.getValueNetworkInputV1_0(testState));
        try {
            MultiLayerNetwork network = MultiLayerNetwork.load(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\src\\main\\resources\\com\\hottes\\caleb\\ultimateticktacktoe\\valueNetworkV1_0.zip"), true);
            network.init();
            System.out.println(testState);
            System.out.println(network.output(Resources.getValueNetworkInputV1_0(testState)).getDouble(0, 0));

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}