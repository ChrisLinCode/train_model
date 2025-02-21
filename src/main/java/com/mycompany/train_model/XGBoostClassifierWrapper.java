package com.mycompany.train_model;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.HashMap;

public class XGBoostClassifierWrapper extends AbstractClassifier {

    private Booster booster;
    private int numClasses;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("Class index is not set!");
        }

        this.numClasses = data.numClasses();

        // 將 Instances 轉換為 XGBoost 的 DMatrix 格式
        DMatrix trainMatrix = convertInstancesToDMatrix(data);

        // 不設置任何參數，讓 XGBoost 使用預設值
        HashMap<String, Object> params = new HashMap<>();
        params.put("objective", numClasses > 2 ? "multi:softprob" : "binary:logistic"); // 僅設置目標函數
        params.put("num_class", numClasses); // 設置類別數（多類分類）

        // 訓練模型
        booster = XGBoost.train(trainMatrix, params, 100, new HashMap<>(), null, null);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // 將單個 Instance 轉換為 XGBoost 格式
        DMatrix testMatrix = convertInstanceToDMatrix(instance);

        // 獲得預測結果
        float[][] predictions = booster.predict(testMatrix);

        // 尋找預測概率最大的類別
        return findMaxIndex(predictions[0]);
    }

    private DMatrix convertInstancesToDMatrix(Instances data) throws XGBoostError {
        int numInstances = data.numInstances();
        int numFeatures = data.numAttributes() - 1;

        float[] featureValues = new float[numInstances * numFeatures];
        float[] labels = new float[numInstances];

        for (int i = 0; i < numInstances; i++) {
            Instance instance = data.instance(i);
            labels[i] = (float) instance.classValue();

            for (int j = 0; j < numFeatures; j++) {
                featureValues[i * numFeatures + j] = (float) instance.value(j);
            }
        }
        // 使用展平的數組構造 DMatrix
        DMatrix dMatrix = new DMatrix(featureValues, numInstances, numFeatures, Float.NaN);
        dMatrix.setLabel(labels);
        return dMatrix;
    }

    private DMatrix convertInstanceToDMatrix(Instance instance) throws XGBoostError {
        int numFeatures = instance.numAttributes() - 1;

        float[] featureValues = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            featureValues[i] = (float) instance.value(i);
        }

        return new DMatrix(featureValues, 1, numFeatures, Float.NaN);
    }

    private int findMaxIndex(float[] probabilities) {
        int maxIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
