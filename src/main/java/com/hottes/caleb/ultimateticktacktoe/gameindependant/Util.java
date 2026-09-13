package com.hottes.caleb.ultimateticktacktoe.gameindependant;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

public class Util {

    public static Map<String, String> constructMapFromFile(String filePath, DupKeyOption dupKeyOption) {
        Map<String, String> map = new HashMap<>();
        try (Stream<String> lines = Files.lines(Paths.get(filePath))) {
            lines.filter(line -> line.contains(":"))
                    .forEach(line -> {
                        String[] keyValuePair = line.split(":", 2);
                        String key = keyValuePair[0];
                        String value = keyValuePair[1];
                        if (DupKeyOption.OVERWRITE == dupKeyOption) {
                            map.put(key, value);
                        } else if (DupKeyOption.DISCARD == dupKeyOption) {
                            map.putIfAbsent(key, value);
                        }
                    });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }

    public static void saveMapToFile(String filepath, Map<String, String> map) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filepath)))) {
            for (String key : map.keySet()) {
                writer.write(key + ":" + map.get(key) + "\n");
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


    }

    enum DupKeyOption {
        OVERWRITE, DISCARD
    }
}
