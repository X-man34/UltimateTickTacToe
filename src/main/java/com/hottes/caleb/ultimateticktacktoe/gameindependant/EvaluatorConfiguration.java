package com.hottes.caleb.ultimateticktacktoe.gameindependant;

import com.hottes.caleb.ultimateticktacktoe.machinelearning.ValueNetworkTrainer;
import org.bytedeco.opencv.presets.opencv_core;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public record EvaluatorConfiguration(double cValue, long maxRolloutDepth, int computeTime, int threads, int stupidity,
                                     boolean allowForcePlay, boolean maxMyCPU, Optional<MultiLayerNetwork> valueNetwork, Optional<MultiLayerNetwork> policyNetwork) {

    private static final String EVALUATOR_CONFIG_NAME = "evaluatorConfig.json";
    private static final String TEMP_DIR = System.getenv("TEMP") + "\\ultimateTickTackToe\\";
    private static final String VALUE_NETWORK_NAME = "value";
    private static final String POLICY_NETWORK_NAME = "policy";
    @Override
    public String toString() {
        return "EvaluatorConfiguration{" +
                "cValue=" + cValue +
                ", maxRolloutDepth=" + maxRolloutDepth +
                ", computeTime=" + computeTime +
                ", threads=" + threads +
                ", stupidity=" + stupidity +
                ", allowForcePlay=" + allowForcePlay +
                ", maxMyCPU=" + maxMyCPU +
                ", valueNetwork=" + (valueNetwork.isPresent()?valueNetwork.get().conf().toJson():"not present") +
                ", policyNetwork=" + (policyNetwork.isPresent()?policyNetwork.get().conf().toJson():"not present") +
                '}';
    }

    /**
     * Save the data nessacary to reconstruct this evaluator into a zip file.
     * The zip file will have in it a json file which is used to deserialize the evaluator object as well as any saved models necessary
     * @param filepath the file of where the evaluator should be saved
     */
    public void saveEvaluator(String filepath) {

        try {
            JSONObject output = new JSONObject();
            output.put("UCB_C", cValue);
            output.put("MaxRolloutDepth", maxRolloutDepth);
            output.put("ComputeTime", computeTime);
            output.put("Threads", threads);
            output.put("Stupidity", stupidity);
            output.put("AllowForcePlay", allowForcePlay);
            output.put("MaxTheCPU", maxMyCPU);
            JSONArray neuralNets = new JSONArray();
            ArrayList<File> filesToZip = new ArrayList<>();
            new File(TEMP_DIR).mkdirs();
            if (valueNetwork.isPresent()) {
                File valueNetworkFile = new File(TEMP_DIR + VALUE_NETWORK_NAME + ".zip");
                valueNetwork.get().save(valueNetworkFile);
                filesToZip.add(valueNetworkFile);
                neuralNets.put(VALUE_NETWORK_NAME);
            }
            if (policyNetwork.isPresent()) {
                File policyNetworkFile = new File(TEMP_DIR + POLICY_NETWORK_NAME + ".zip");
                policyNetwork.get().save(policyNetworkFile);
                filesToZip.add(policyNetworkFile);
                neuralNets.put(POLICY_NETWORK_NAME);
            }
            //could also save the policy network when thats made.

            output.put("NeuralNets", neuralNets);
            String jsonConfigFileTempOut = TEMP_DIR + EVALUATOR_CONFIG_NAME;
            PrintStream jsonOut = new PrintStream(jsonConfigFileTempOut);
            jsonOut.println(output.toString(5));
            jsonOut.close();
            filesToZip.add(new File(jsonConfigFileTempOut));

            ZipOutputStream zipOut = new ZipOutputStream(new FileOutputStream(filepath));


            for (File fileToZip : filesToZip) {
                FileInputStream fis = new FileInputStream(fileToZip);
                ZipEntry zipEntry = new ZipEntry(fileToZip.getName());
                zipOut.putNextEntry(zipEntry);

                byte[] bytes = new byte[1024];
                int length;
                while((length = fis.read(bytes)) >= 0) {
                    zipOut.write(bytes, 0, length);
                }
                fis.close();
            }

            zipOut.close();
            new File(TEMP_DIR).delete();

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public static EvaluatorConfiguration getInstance(File saveZipFile) throws IOException {

        //unzip the file
        ZipInputStream zipInputStream = new ZipInputStream(new FileInputStream(saveZipFile));
        ZipEntry zipEntry;
        System.out.println(TEMP_DIR);
        while ((zipEntry = zipInputStream.getNextEntry()) != null) {
            File newFile = new File(TEMP_DIR, zipEntry.getName());

            // Create directories for subdirectories in zip
            if (zipEntry.isDirectory()) {
                newFile.mkdirs();
            } else {
                // Make sure the parent directory exists
                new File(newFile.getParent()).mkdirs();

                try (FileOutputStream fos = new FileOutputStream(newFile)) {
                    int len;
                    byte[] buffer = new byte[1024];
                    while ((len = zipInputStream.read(buffer)) > 0) {
                        fos.write(buffer, 0, len);
                    }
                }
            }
            zipInputStream.closeEntry();



        }
        //read the json and create the evaluator config
        FileInputStream jsonIN = new FileInputStream(TEMP_DIR + EVALUATOR_CONFIG_NAME);
        JSONObject jsonObject = new JSONObject(new String(jsonIN.readAllBytes()));
        jsonIN.close();
        JSONArray neuralNets = jsonObject.optJSONArray("NeuralNets", new JSONArray());
        Optional<MultiLayerNetwork> valueNetwork = Optional.empty();
        Optional<MultiLayerNetwork> policyNetwork = Optional.empty();
        //try and load any neural networks that were specified
        for (int i = 0; i < neuralNets.length(); i++) {
            String networkName = neuralNets.getString(i);

            switch (networkName) {
                case "value":
                    valueNetwork = Optional.of(MultiLayerNetwork.load(new File(TEMP_DIR + networkName + ".zip"), false));
                case "policy":
                    policyNetwork = Optional.of(MultiLayerNetwork.load(new File(TEMP_DIR + networkName + ".zip"), false));
            }
        }
        new File(TEMP_DIR).delete();

        return new EvaluatorConfiguration(jsonObject.optDouble("UCB_C", 2),
                jsonObject.optInt("MaxRolloutDepth", 1000),
                jsonObject.optInt("ComputeTime", 30),
                jsonObject.optInt("Threads", 2),
                jsonObject.optInt("Stupidity", 0),
                jsonObject.optBoolean("AllowForcePlay", false),
                jsonObject.optBoolean("MaxTheCPU", false),
                valueNetwork,
                policyNetwork);
    }
}
