module com.hottes.caleb.ultimateticktacktoe {
   // requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires jdk.incubator.vector;
    requires javafx.swing;
    requires java.sql;
    requires java.instrument;
    requires org.junit.platform.suite.api;
    requires jol.core;

    opens com.hottes.caleb.ultimateticktacktoe to javafx.fxml;
    exports com.hottes.caleb.ultimateticktacktoe;
    exports com.hottes.caleb.ultimateticktacktoe.ui;
    opens com.hottes.caleb.ultimateticktacktoe.ui to javafx.fxml;
}