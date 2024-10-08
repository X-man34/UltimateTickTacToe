package com.hottes.caleb.ultimateticktacktoe;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.util.*;
import java.util.stream.Stream;

import static com.hottes.caleb.ultimateticktacktoe.Resources.generateMatrixPermutations;
import static org.junit.jupiter.api.Assertions.*;

class BoardStateTest {

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
            true, 3);

    private final SubBoardState inProgress = new SubBoardState(new double[][]{
            {1,0,1},
            {0,-1,0},
            {1,0,0}},
            true, 3);

    private final SubBoardState draw = new SubBoardState(new double[][]{
            {1,-1,1},
            {1,-1,-1},
            {-1,1,1}},
            false, 3);

    private BoardState emptyState;
    private BoardState oneWinTopRowState;
    private BoardState twoWinTopRowState;
    private BoardState oneWinLeftSide;
    private BoardState twoWinLeftSide;
    private BoardState oneWinMainDiagonal;
    private BoardState twoWinMainDiagonal;
    private BoardState oneWinOffDiagonal;
    private BoardState twoWinOffDiagonal;
    private BoardState inprogressGame;
    private BoardState drawState;

    @BeforeEach()
    void setUpTest() {
        emptyState = new BoardState(new SubBoardState[][]{
                {empty, empty, empty},
                {empty, empty, empty},
                {empty, empty, empty}},
                3);
        emptyState.setAllBoardsActivity(true);
        oneWinTopRowState = new BoardState(new SubBoardState[][]{
                {oneWin, oneWin, oneWin},
                {draw, inProgress, empty},
                {empty, empty, empty}},
                3);
        oneWinTopRowState.setAllBoardsActivity(false);
        twoWinTopRowState = new BoardState(new SubBoardState[][]{
                {twoWin, twoWin, twoWin},
                {inProgress, empty, empty},
                {empty, draw, inProgress}},
                3);
        twoWinTopRowState.setAllBoardsActivity(false);
        oneWinLeftSide = new BoardState(new SubBoardState[][]{
                {oneWin, draw, inProgress},
                {oneWin, empty, empty},
                {oneWin, inProgress, empty}},
                3);
        oneWinLeftSide.setAllBoardsActivity(false);
        twoWinLeftSide = new BoardState(new SubBoardState[][]{
                {twoWin, inProgress, empty},
                {twoWin, draw, empty},
                {twoWin, empty, inProgress}},
                3);
        twoWinLeftSide.setAllBoardsActivity(false);
        oneWinMainDiagonal = new BoardState(new SubBoardState[][]{
                {oneWin, empty, empty},
                {empty, oneWin, draw},
                {empty, inProgress, oneWin}},
                3);
        oneWinMainDiagonal.setAllBoardsActivity(false);
        twoWinMainDiagonal = new BoardState(new SubBoardState[][]{
                {twoWin, empty, empty},
                {empty, twoWin, empty},
                {draw, empty, twoWin}},
                3);
        twoWinMainDiagonal.setAllBoardsActivity(false);
        oneWinOffDiagonal = new BoardState(new SubBoardState[][]{
                {empty, inProgress, oneWin},
                {empty, oneWin, draw},
                {oneWin, inProgress, empty}},
                3);
        oneWinOffDiagonal.setAllBoardsActivity(false);
        twoWinOffDiagonal = new BoardState(new SubBoardState[][]{
                {empty, draw, twoWin},
                {inProgress, twoWin, empty},
                {twoWin, empty, empty}},
                3);
        twoWinOffDiagonal.setAllBoardsActivity(false);
        inprogressGame = new BoardState(new SubBoardState[][]{
                {oneWin, empty, oneWin},
                {inProgress, twoWin, empty},
                {oneWin, draw, empty}},
                3);
        inprogressGame.setAllBoardsActivity(false);
        inprogressGame.setBoardActive(1, 0);
        drawState = new BoardState(new SubBoardState[][]{
                {oneWin, twoWin, oneWin},
                {oneWin, twoWin, twoWin},
                {twoWin, oneWin, oneWin}},
                3);
        drawState.setAllBoardsActivity(false);
    }
    @Test
    void getEvaluation() {
        assertEquals(Resources.Evaluation.IN_PROGRESS.getlabel(), emptyState.getEvaluation(), "Emtpy board does not evaluate to in progress");
        assertEquals(Resources.Evaluation.PLAYER_ONE_WIN.getlabel(), oneWinTopRowState.getEvaluation(), "Player one win upper row not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_TWO_WIN.getlabel(), twoWinTopRowState.getEvaluation(), "Player two win upper row not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_ONE_WIN.getlabel(), oneWinLeftSide.getEvaluation(), "Player one win left side not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_TWO_WIN.getlabel(), twoWinLeftSide.getEvaluation(), "Player two win left side not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_ONE_WIN.getlabel(), oneWinMainDiagonal.getEvaluation(), "Player one win main diagonal not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_TWO_WIN.getlabel(), twoWinMainDiagonal.getEvaluation(), "Player two win main diagonal not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_ONE_WIN.getlabel(), oneWinOffDiagonal.getEvaluation(), "Player one win off diagonal not evaluated correctly");
        assertEquals(Resources.Evaluation.PLAYER_TWO_WIN.getlabel(), twoWinOffDiagonal.getEvaluation(), "Player two win off diagonal not evaluated correctly");
        assertEquals(Resources.Evaluation.IN_PROGRESS.getlabel(), inprogressGame.getEvaluation(), "Arbitrary in progress game not evaluated correctly");
        assertEquals(Resources.Evaluation.DRAW.getlabel(), drawState.getEvaluation(), "Draw not evaluated correctly");

        
    }

    @Test
    @Tag("benchmark")
    void benchmarkEvaluationFunction() {
        System.out.println("=======================\nBenchmarking BoardState Evaluation Function");
        List<double[][]> rawPermutations = new ArrayList<>();
        generateMatrixPermutations(new double[3][3], 0, 0, rawPermutations);
        List<SubBoardState> ticTacToeStates = new ArrayList<>();
        rawPermutations.forEach(doubles -> ticTacToeStates.add(new SubBoardState(doubles, false, 3)));

        ArrayList<BoardState> boardStates = new ArrayList<>();
        for (int k = 0; k < 50; k++) {
            Collections.shuffle(ticTacToeStates);
            for (int i = 0; i < Math.floorDivExact(ticTacToeStates.size(), 9); i++) {
                for (int j = 0; j < 9; j++) {
                    boardStates.add(new BoardState(new SubBoardState[][]{
                            {ticTacToeStates.get((i * 9) + j), ticTacToeStates.get((i * 9) + j), ticTacToeStates.get((i * 9) + j)},
                            {ticTacToeStates.get((i * 9) + j), ticTacToeStates.get((i * 9) + j), ticTacToeStates.get((i * 9) + j)},
                            {ticTacToeStates.get((i * 9) + j), ticTacToeStates.get((i * 9) + j), ticTacToeStates.get((i * 9) + j)}},

                            3));

                }

            }
        }

        //by now we have 2187 * k random board states which we can use to do a preformance test on the board state evaluation function.
        //if k is three then the basic 3x3 eval function will be called around 60561 times.
        long startTime = System.currentTimeMillis();
        for (BoardState state : boardStates) {
            state.getEvaluation();
        }
        long endTime = System.currentTimeMillis();
        long time = endTime - startTime;
        double averageTime = (double) time / boardStates.size() * 1000;//us/eval call
        System.out.println("Took: " + time + " ms to run " + boardStates.size() + " board state evaluations");
        System.out.println("Average BoardStateEvaluation time: " + averageTime + " microseconds");
        //1.2 million iterations will accure around 42 million calls to this function.
        System.out.println("Benchmarked BoardState Evaluation Function\n=======================");
    }


    @Test
    void getActions() {
        inprogressGame.setPlayerOneTurn(true);
        ArrayList<GameAction> actions = inprogressGame.getActions();
        ArrayList<GameAction> expectedActions = new ArrayList<>();
        expectedActions.add(new UltimateTickTacToeGameAction(1, 0, 0, 1, 1));
        expectedActions.add(new UltimateTickTacToeGameAction(1, 0, 1, 0, 1));
        expectedActions.add(new UltimateTickTacToeGameAction(1, 0, 1, 2, 1));
        expectedActions.add(new UltimateTickTacToeGameAction(1, 0, 2, 1, 1));
        expectedActions.add(new UltimateTickTacToeGameAction(1, 0, 2, 2, 1));
        assertEquals(expectedActions.size(), actions.size(), "Incorrect number of actions created");
        assertArrayEquals(expectedActions.toArray(), actions.toArray(), "Generated actions not correct. ");


    }
    @Test
    @Disabled
    void preformAction() {

    }

    @Test
    void simulateAction() {
        BoardState newState = (BoardState) inprogressGame.simulateAction(new UltimateTickTacToeGameAction(1, 0, 0, 1, 0), false);//simulate an action that does nothing
        newState.preformAction(new UltimateTickTacToeGameAction(1, 0, 2, 1, 1));//make a change. If simulate action returns a copy like it is supposed to then this change should not be reflected in the original state
        assertNotEquals(inprogressGame.getMinorBoardAt(1, 0).itemAt(2,1), newState.getMinorBoardAt(1, 0).itemAt(2,1), "New state is pointer to old state");//preforms an action which should not actualy change the state of the board
        //if we got this far then inprogressGame has not actually been affected.
        GameAction action = new UltimateTickTacToeGameAction(1, 0, 2, 2, -1);
        newState = (BoardState) inprogressGame.simulateAction(action, false);
        assertEquals(newState.getMinorBoardAt(1, 0).getStateAsCopy()[action.y][action.x], action.getMarker(), "New state does not contain marker in correct place");
        BoardState invertedState =  (BoardState) inprogressGame.simulateAction(action, true);
        assertEquals(invertedState.getMinorBoardAt(1, 0).getStateAsCopy()[action.y][action.x], -action.getMarker(), "Failed to invert board");
    }

    @Test
    @Disabled
    void setAllBoardsActivity() {
    }

    @Test
    @Disabled
    void setBoardActive() {
    }


    @Test
    void testHashStringConstructor() {
        BoardState testState = new BoardState(3);
        testState.preformAction(new UltimateTickTacToeGameAction(2, 1, 1, 1, -1));
        testState.preformAction(new UltimateTickTacToeGameAction(1, 0, 0, 0, 1));
        testState.preformAction(new UltimateTickTacToeGameAction(0, 2, 1, 2, -1));
        testState.preformAction(new UltimateTickTacToeGameAction(0, 0, 2, 1, 1));
        BoardState newState = new BoardState(testState.getStringHash());
        assertEquals(testState.getStringHash(), newState.getStringHash(), "String hashes were not equal");
        assertEquals(testState, newState,"States were not equal");

        System.out.println(testState);
        System.out.println(Arrays.toString(testState.getBitSet().toByteArray()));
        System.out.println(Resources.getBitSetAsString(testState.getBitSet()));

    }

    @Test
    void testgetBitSet() {
        BoardState testState = new BoardState(3);
        testState.preformAction(new UltimateTickTacToeGameAction(2, 1, 1, 1, -1));
        testState.preformAction(new UltimateTickTacToeGameAction(1, 0, 0, 0, 1));
        testState.preformAction(new UltimateTickTacToeGameAction(0, 2, 1, 2, -1));
        testState.preformAction(new UltimateTickTacToeGameAction(0, 0, 2, 1, 1));
        testState.setAllBoardsActivity(false);
        testState.setBoardActive(2, 1);
        BitSet actualSet = BitSet.valueOf(new byte[] {3, 1, 2, 4, 0, 0, 32, 0, 8, 16, 0, 0, 0, 0, 0, 0, 0, 64, 8});
        assertEquals(actualSet, testState.getBitSet(), "Returned bit set was not correct for simple state");

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
        testState = new BoardState(new SubBoardState[][]{
                {oneWin, oneWin, topRight},
                {oneWin, twoWin, activeBoard},
                {bottomLeft, empty, empty}},
                3);
        testState.setAllBoardsActivity(false);
        testState.setBoardActive(1, 2);
        actualSet = BitSet.valueOf(new byte[] {3, 29, 56, -32, -64, 1, -83, 1, 59, 112, -64, 1, 0, -127, 1, 35, 65, 2});
        assertEquals(actualSet, testState.getBitSet(), "Returned bit set was not correct for X play so O can win");
    }


    @Test
    void testBitSetConstructor() {
        BoardState actualState = new BoardState(3);
        actualState.preformAction(new UltimateTickTacToeGameAction(2, 1, 1, 1, -1));
        actualState.preformAction(new UltimateTickTacToeGameAction(1, 0, 0, 0, 1));
        actualState.preformAction(new UltimateTickTacToeGameAction(0, 2, 1, 2, -1));
        actualState.preformAction(new UltimateTickTacToeGameAction(0, 0, 2, 1, 1));
        actualState.setAllBoardsActivity(false);
        actualState.setBoardActive(2, 1);
        actualState.setPlayerOneTurn(false);

        BoardState testState = new BoardState(BitSet.valueOf(new byte[] {3, 0, 2, 4, 0, 0, 32, 0, 8, 16, 0, 0, 0, 0, 0, 0, 0, 64, 8}));
        assertTwoBoardStatesEqual(actualState, testState, "Board state was incorrectly constructed from bit set for simple position");

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
        actualState = new BoardState(new SubBoardState[][]{
                {oneWin, oneWin, topRight},
                {oneWin, twoWin, activeBoard},
                {bottomLeft, empty, empty}},
                3);
        actualState.setAllBoardsActivity(false);
        actualState.setBoardActive(1, 2);
        actualState.setPlayerOneTurn(true);
        testState = new BoardState(BitSet.valueOf(new byte[] {3, 29, 56, -32, -64, 1, -83, 1, 59, 112, -64, 1, 0, -127, 1, 35, 65, 2}));
        assertTwoBoardStatesEqual(actualState, testState, "Board state was incorrectly constructed from bit set for X play so O can win");

    }

    @Test
    void testBoardStateClone() {

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
        BoardState originalState = new BoardState(new SubBoardState[][]{
                {oneWin, oneWin, topRight},
                {oneWin, twoWin, activeBoard},
                {bottomLeft, empty, empty}},
                3);
        originalState.setAllBoardsActivity(false);
        originalState.setBoardActive(1, 2);
        originalState.setPlayerOneTurn(false);
        BoardState newState = originalState.getClone();
        originalState.togglePlayerOneTurn();
        newState.togglePlayerOneTurn();
        assertTwoBoardStatesEqual(originalState, newState, "Clone state not the same as original");
        newState.togglePlayerOneTurn();
        assertTwoBoardStatesNotEqual(originalState, newState, "Clone state has mutable player one turn");
        newState.togglePlayerOneTurn();
        newState.setAllBoardsActivity(false);
        assertTwoBoardStatesNotEqual(originalState, newState, "Clone state has mutable board activity");
        originalState.setAllBoardsActivity(false);
        newState.preformAction(new UltimateTickTacToeGameAction(1, 2, 2, 2, 1));
        assertTwoBoardStatesNotEqual(originalState, newState, "Clone state has mutable state");

    }



    private void assertTwoBoardStatesNotEqual(BoardState actualState, BoardState testState, String baseMessage) {
        boolean testPassed = false;
        try {
            assertTwoBoardStatesEqual(actualState, testState, baseMessage);
        }catch (AssertionError e) {
            //if an exception is caught then something failed, which is actually success.
            testPassed = true;
            baseMessage = e.getMessage();
        }
        assertTrue(testPassed, baseMessage);
    }
    private void assertTwoBoardStatesEqual(BoardState actualState, BoardState testState, String baseMessage) {
        assertEquals(actualState, testState, baseMessage);

        ArrayList<Executable> activityTests = new ArrayList<>();
        ArrayList<Executable> boardStateTests = new ArrayList<>();
        assertEquals(actualState.getRows(), testState.getRows(), baseMessage + ", rows not the same");
        assertEquals(actualState.getCols(), testState.getCols(), baseMessage + ", columsn not the same");

        for (int i = 0; i < actualState.getRows(); i++) {
            for (int j = 0; j < actualState.getCols(); j++) {

                int finalI = i;
                int finalJ = j;
                activityTests.add(() -> assertEquals(actualState.getMinorBoardAt(finalI, finalJ).isActive(), testState.getMinorBoardAt(finalI, finalJ).isActive(), "Sub board: " + finalI + " " + finalJ + " did not have matching activity"));
                boardStateTests.add(() -> assertTrue(Arrays.deepEquals(actualState.getMinorBoardAt(finalI, finalJ).getStateAsCopy(), testState.getMinorBoardAt(finalI, finalJ).getStateAsCopy()), "Sub board: " + finalI + " " + finalJ + " did not have matching state expected:\n" + actualState.getMinorBoardAt(finalI, finalJ) + "actual: \n" + testState.getMinorBoardAt(finalI, finalJ)));

            }
        }
        assertAll(boardStateTests);
        assertEquals(actualState.isPlayerOneTurn(), testState.isPlayerOneTurn(), baseMessage + ", whose turn it was not set correctly");

        assertAll(activityTests);
    }


}