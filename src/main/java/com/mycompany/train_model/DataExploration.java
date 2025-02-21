package com.mycompany.train_model;

import weka.core.Instances;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class DataExploration {

    // 類別分布統計
    public static void analyzeClassDistribution(Instances data, String outputFilePath) {
        Map<String, Integer> classCounts = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            String classValue = data.instance(i).stringValue(data.classIndex());
            classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath, true))) {
            writer.write("Class Distribution:\n");
            classCounts.forEach((key, value) -> {
                try {
                    writer.write(key + ": " + value + "\n");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
            writer.write("\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 文本長度分布統計
    public static void analyzeTextLength(Instances data, String outputFilePath) {
        List<Integer> lengths = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            String text = data.instance(i).stringValue(0);
            lengths.add(text.length());
        }
        double avgLength = lengths.stream().mapToInt(Integer::intValue).average().orElse(0);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath, true))) {
            writer.write("Average Text Length: " + avgLength + "\n\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 檢查遺漏值
    public static void checkMissingValues(Instances data, String outputFilePath) {
        int missingCount = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (data.instance(i).isMissing(j)) {
                    missingCount++;
                }
            }
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath, true))) {
            writer.write("Number of Missing Values: " + missingCount + "\n\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
