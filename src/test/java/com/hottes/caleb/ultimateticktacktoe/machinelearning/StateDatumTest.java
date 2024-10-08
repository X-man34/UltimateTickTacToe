package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class StateDatumTest {

    @Test
    void getIndex() {
        assertEquals(0, StateDatum.getIndex(0, 0, 0, 0));
        assertEquals(4, StateDatum.getIndex(0, 0, 1, 1));
        assertEquals(37, StateDatum.getIndex(1, 1, 0, 1));
        assertEquals(69, StateDatum.getIndex(2, 1, 2, 0));
        assertEquals(80, StateDatum.getIndex(2, 2, 2, 2));
        assertEquals(20, StateDatum.getIndex(0, 2, 0, 2));
        assertEquals(60, StateDatum.getIndex(2, 0, 2, 0));
        assertEquals(49, StateDatum.getIndex(1, 2, 1, 1));


    }
}