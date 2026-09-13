package com.hottes.caleb.ultimateticktacktoe.gameindependant;

import java.awt.*;

/**
 * Represents an action that a user can take, basically a fancy point
 * the Marker field represents what is going to be put in the state array when the move is made or simulated.
 * It should probably be positive as the current player should be the one playing
 */
public class GameAction extends Point {

    private double marker;

    /**
     * Constructs a game action with all fields speficifed
     *
     * @param x
     * @param y
     * @param theMarker
     */
    public GameAction(int x, int y, double theMarker) {
        super(x, y);
        this.marker = theMarker;
    }

    /**
     * Constructs a Game Action from another Game Action
     *
     * @param action
     */
    public GameAction(GameAction action) {
        this(action.x, action.y, action.marker);
    }

    /**
     * Constructs a Game Action with a marker of 1
     *
     * @param x
     * @param y
     */
    public GameAction(int x, int y) {
        this(x, y, 1);
    }

    public double getMarker() {
        return marker;
    }

    public void setMarker(double marker) {
        this.marker = marker;
    }

    public void invertMarker() {
        marker = -marker;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        GameAction action = (GameAction) o;
        return java.lang.Double.compare(marker, action.marker) == 0;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + java.lang.Double.hashCode(marker);
        return result;
    }
}
