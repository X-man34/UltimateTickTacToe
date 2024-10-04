package com.hottes.caleb.ultimateticktacktoe.ui;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;

/**
 * Represents an action on a ultimate tick tack toe board. The major board coords are saved in the superclass x and y variables but to preform this action one would need all the information
 */
public class UltimateTickTacToeGameAction extends GameAction {

    public final int majorRow;
    public final int majorCol;

    /**
     *
     * @param majorRow the row of the board being played on
     * @param majorCol the column of the board being played on
     * @param minorRow the row of the square of the board being played on
     * @param minorCol the column of the square of the board being played on
     * @param theMarker the marker of the player playing
     */
    public UltimateTickTacToeGameAction(int majorRow, int majorCol, int minorRow, int minorCol, double theMarker) {
        super(minorCol, minorRow, theMarker);
        this.majorRow = majorRow;
        this.majorCol = majorCol;
    }

    public int getMajorCol() {
        return majorCol;
    }

    public int getMajorRow() {
        return majorRow;
    }

    @Override
    public String toString() {
        return "Action{" +
                "majR=" + majorRow +
                ", majC=" + majorCol +
                ", minR=" + y +
                ", minC=" + x +
                ", marker=" + getMarker() +
                '}';
    }
}
