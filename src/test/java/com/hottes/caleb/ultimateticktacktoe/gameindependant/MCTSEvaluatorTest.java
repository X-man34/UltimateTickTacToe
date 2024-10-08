package com.hottes.caleb.ultimateticktacktoe.gameindependant;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.SubBoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts.NodeData;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTree;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTreeNode;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import javafx.application.Platform;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;


import static org.junit.jupiter.api.Assertions.*;

class MCTSEvaluatorTest {

    private final SubBoardState oneWin = new SubBoardState(new double[][]{
            {1,1,1},
            {0,0,0},
            {0,0,0}},
            false, 3);

    private final SubBoardState twoWin = new SubBoardState(new double[][]{
            {-1,-1,-1},
            {0,0,0},
            {0,0,0}},
            false, 3);

    private final SubBoardState empty = new SubBoardState(new double[][]{
            {0,0,0},
            {0,0,0},
            {0,0,0}},
            false, 3);

    private final SubBoardState inProgress = new SubBoardState(new double[][]{
            {1,-1,1},
            {0,-1,0},
            {1,1,-1}},
            true, 3);

    private final SubBoardState draw = new SubBoardState(new double[][]{
            {1,-1,1},
            {1,-1,-1},
            {-1,1,1}},
            false, 3);
    @Test
    void oneMoveToWin() {
        //set up a state where it is player 1's turn and they can win by playing in a specific place and player -1 has no hope of winning. The best move is obvious.
        BoardState testState = new BoardState(new SubBoardState[][]{
                {oneWin, empty, empty},
                {oneWin, empty, empty},
                {inProgress, empty, empty}},
                3);
        testState.setPlayerOneTurn(true);
        testState.setAllBoardsActivity(false);
        testState.setBoardActive(2, 0);

        MCTSEvaluator evaluator = new MCTSEvaluator(testState);
        for (int i = 0; i < 30; i++) {
            evaluator.preformIteration(evaluator.tree.getRoot());
        }
        GameAction botAction = evaluator.getCurrentBestMove();
        assertEquals(new UltimateTickTacToeGameAction(2, 0, 1, 0, 1), botAction, "Evaluator failed the following puzzle: {1 to play, activeboardRow=2, col=0}" + testState);
    }

    //if this test fails try increasing the number of iterations
    @Test
    void forceOPlaySoXCanWin() {

        SubBoardState topRight = new SubBoardState(new double[][]{
                {-1,0,-1},
                {-1,0,-1},
                {0,1,1}},
                false, 3);
        SubBoardState activeBoard = new SubBoardState(new double[][]{
                {0,0,0},
                {0,0,0},
                {1,1,0}},
                false, 3);
        SubBoardState bottomLeft = new SubBoardState(new double[][]{
                {0,1,0},
                {0,1,0},
                {0,0,0}},
                false, 3);
        BoardState testState = new BoardState(new SubBoardState[][]{
                {oneWin, oneWin, topRight},
                {oneWin, twoWin, activeBoard},
                {bottomLeft, empty, empty}},
                3);
        testState.setAllBoardsActivity(false);
        testState.setBoardActive(1, 2);
        testState.setPlayerOneTurn(true);

        MCTSEvaluator evaluator = new MCTSEvaluator(testState);
        for (int i = 0; i < 40000; i++) {
            evaluator.preformIteration(evaluator.tree.getRoot());

        }

        GameAction botAction = evaluator.getCurrentBestMove();
        assertEquals(new UltimateTickTacToeGameAction(1, 2, 0, 2, 1), botAction, "Evaluator failed the following puzzle: {1 to play, activeboardRow=1, col=2}" + testState);

    }



    private MCTSEvaluator mcts;
    private BoardState emptyBoard;

    @BeforeEach
    void setUp() {
        emptyBoard = new BoardState(new SubBoardState[][]{
                {empty, empty, empty},
                {empty, empty, empty},
                {empty, empty, empty}},
                3);
        emptyBoard.setAllBoardsActivity(true);
        mcts = new MCTSEvaluator(emptyBoard);
    }

    @Test
    void testConstructor_shouldSetRootWithInitialState() {
        // Then
        assertNotNull(mcts.tree.getRoot());
        assertEquals(emptyBoard, mcts.tree.getRoot().getData().getGameState());
    }



    @Test
    void testPreformIteration_shouldExpandTree() {
        // Given

        // When
        mcts.preformIteration(mcts.tree.getRoot());

        // Then
        assertEquals(81, mcts.tree.getRoot().getNumberOfChildren());//changed from 1 to 81
    }

    @Test
    void testPreformIteration_shouldVisitedLeafNode() {
        // Given

        // When
        mcts.preformIteration(mcts.tree.getRoot());
        GenericTreeNode<NodeData> leafNode = mcts.tree.getRoot().getChildAt(0);

        // Then
        assertEquals(1, leafNode.getData().getNumVisits());
    }


    @Test
    void testGetUCB1_shouldReturnPositiveInfinityForUnvisitedNode() {
        // Given
        NodeData unvisitedNodeData = new NodeData(0, 0, emptyBoard, null);
        GenericTreeNode<NodeData> unvisitedNode = new GenericTreeNode<>(unvisitedNodeData);

        // When
        double ucb1 = mcts.getUCB1(unvisitedNode);

        // Then
        assertEquals(Double.POSITIVE_INFINITY, ucb1);
    }

    @Test
    @Tag("benchmark")
    void benchmarkIterationSpeed() {
        System.out.println("===================\nBenchmarking iteration speed");

        long preTreeHeapFreeMemory = Runtime.getRuntime().totalMemory();
        BoardState initialState = new BoardState(3);
        initialState.preformAction(new UltimateTickTacToeGameAction(1, 1, 1,1, 1));//this is not an arbirary move but a strategic one, giving the AI lots of options to think about and is also the best move X can take

        MCTSEvaluator evaluator = new MCTSEvaluator(initialState, new EvaluatorConfiguration(2, 1000, 60, 5, 100, false, false));//most of these settings don't matter we will preform the search ourselves in this test.

        long numIterations = 10000;//can't do too many or it will run out of heap space.
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < numIterations; i++) {
            evaluator.preformIteration(evaluator.tree.getRoot());
        }
        long totalTime = System.currentTimeMillis() - startTime;
        double treeMemory = (double) (preTreeHeapFreeMemory - Runtime.getRuntime().freeMemory()) / 1000000000;
        double averageTime = (double) totalTime / numIterations;
        int iterationsIn60Secs = (int) Math.round(60000/averageTime);
        double memoryPerIteration = treeMemory * 1000000 / numIterations;
        System.out.println("Took: " + totalTime + " ms to run " + numIterations + " iterations");
        System.out.println("Average Time was: " + averageTime + " ms");
        System.out.println("At this rate in 1 minute: " + iterationsIn60Secs + " iterations can be run");
        System.out.println("Tree used: " + treeMemory + " GB of memory");
        System.out.println("Average: " + memoryPerIteration + " KB used per iteration");
        System.out.println("Done Benchmarking iteration speed\n===========================");
        //if the test fails it is likly becuase of an out of memory issue.
    }

//    @Test
//    @Tag("benchmark")
//    void analyzeMemory() {
//        System.out.println("================\ntesting memory");
//        BoardState initialState = new BoardState(3);
//        initialState.preformAction(new UltimateTickTacToeGameAction(1, 1, 1,1, 1));//this is not an arbirary move but a strategic one, giving the AI lots of options to think about and is also the best move X can take
//        String initalStateHash = initialState.getStringHash();
//        System.out.println(GraphLayout.parseInstance(initalStateHash).toFootprint());
//        System.out.println(GraphLayout.parseInstance(initialState).toFootprint());
//        MCTSEvaluator evaluator = new MCTSEvaluator(initialState, new EvaluatorConfiguration(2, 1000, 60, 5, 100, false, false));//most of these settings don't matter we will preform the search ourselves in this test.
//        evaluator.preformIteration(evaluator.tree.getRoot());
//
//        System.out.println(GraphLayout.parseInstance(evaluator).toFootprint());
//        System.out.println("done testing memory\n=====================");
//    }
}