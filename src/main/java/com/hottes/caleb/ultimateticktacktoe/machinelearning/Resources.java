package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.hottes.caleb.ultimateticktacktoe.Resources.getIndex;

public class Resources {


    /**
     * Takes a filepath and series name and puts all data files with a specific prefix eg value or policy
     *
     * @param dataDir    the source of the data storage folder
     * @param seriesName the name of the data series
     * @return the dataset
     */
    public static DataSet concatenateValueDataFiles(File dataDir, String seriesName, String prefix) throws FileNotFoundException {
        if (!dataDir.isDirectory()) {
            throw new IllegalArgumentException("Provided file is not a folder. ");
        }
        File seriesDir = new File(dataDir.getAbsolutePath() + "/" + seriesName);
        if (!seriesDir.isDirectory()) {
            throw new FileNotFoundException("Data folder for series: " + seriesName + " not found.");
        }
        List<DataSet> dataSets = new ArrayList<>();
        for (File file : Objects.requireNonNull(seriesDir.listFiles((dir, name) -> name.endsWith(".bin") && name.startsWith(prefix)))) {
            DataSet dataSet = new DataSet();
            dataSet.load(file);
            dataSets.add(dataSet);
        }
        return concatenateDataSets(dataSets);

    }


    private static DataSet concatenateDataSets(List<DataSet> dataSets) {
        // Get total number of features and labels
        INDArray features = dataSets.get(0).getFeatures();
        INDArray labels = dataSets.get(0).getLabels();

        for (int i = 1; i < dataSets.size(); i++) {
            features = Nd4j.vstack(features, dataSets.get(i).getFeatures());
            labels = Nd4j.vstack(labels, dataSets.get(i).getLabels());
        }

        return new DataSet(features, labels);
    }

    static int getActivityIndex(int majRow, int majCol, int boardSize) {
        return getIndex(boardSize - 1, boardSize - 1, boardSize - 1, boardSize - 1, boardSize) + 1 + (majRow * boardSize) + majCol;
    }
}
