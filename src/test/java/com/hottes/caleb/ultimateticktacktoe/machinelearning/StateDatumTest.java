package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.Resources;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class StateDatumTest {

    @Test
    void getIndex() {
        assertEquals(0, Resources.getIndex(0, 0, 0, 0,3 ));
        assertEquals(4, Resources.getIndex(0, 0, 1, 1,3 ));
        assertEquals(37,Resources.getIndex(1, 1, 0, 1,3 ));
        assertEquals(69,Resources.getIndex(2, 1, 2, 0,3 ));
        assertEquals(80,Resources.getIndex(2, 2, 2, 2,3 ));
        assertEquals(20,Resources.getIndex(0, 2, 0, 2,3 ));
        assertEquals(60,Resources.getIndex(2, 0, 2, 0,3 ));
        assertEquals(49,Resources.getIndex(1, 2, 1, 1,3 ));


    }
}