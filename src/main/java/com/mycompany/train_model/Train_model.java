package com.mycompany.train_model;

import com.google.gson.Gson;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;
import edu.stanford.nlp.pipeline.*;
import java.util.concurrent.*;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.Resample;
import weka.filters.Filter;

//演算法
import weka.classifiers.functions.Logistic;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.MultilayerPerceptron;

import com.github.jfasttext.JFastText;

public class Train_model {

    private static StanfordCoreNLP pipeline;
    private static JFastText fastText;
    private static final Set<String> stopWords = new HashSet<>();

    static {
        initializePipeline();
        loadStopWords();
    }
    
    public static class PredictionAPI {

        private UnifiedModel model;

        // 修改後的建構子接收兩個參數
        public PredictionAPI(String unifiedModelPath, String fastTextModelPath) throws IOException, ClassNotFoundException {
            this.model = UnifiedModel.loadModel(unifiedModelPath);
            this.model.setFastText(loadFastTextModel(fastTextModelPath));
        }

        public String predict(String inputText, int quantity) throws Exception {
            return model.predict(inputText, quantity);//預測
        }

        public String predictClass(String inputText) throws Exception {
            return model.predictClass(inputText); // 貼標
        }
    }


    private static Instances vectorizeData(Instances data, JFastText fastText) {
        //int vectorSize = 300;
        int vectorSize = fastText.getVector("示例文本").size(); 
        ArrayList<Attribute> attributes = new ArrayList<>();

        // 定義每個向量維度的屬性
        for (int i = 0; i < vectorSize; i++) {
            attributes.add(new Attribute("vec_" + i));
        }
        attributes.add(data.classAttribute());

        Instances vectorizedData = new Instances("VectorizedData", attributes, data.numInstances());
        vectorizedData.setClassIndex(vectorizedData.numAttributes() - 1);

        // 用於儲存已計算過的詞向量的緩存
        ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();

        for (int i = 0; i < data.numInstances(); i++) {
            double[] vector = new double[vectorSize];
            String text = data.instance(i).stringValue(0);
            String[] tokens = text.split(" ");
            int count = 0;

            for (String token : tokens) {
                double[] wordVector;
                if (vectorCache.containsKey(token)) {
                    wordVector = vectorCache.get(token);
                } else {
                    List<Float> ftVectorList = fastText.getVector(token);
                    if (ftVectorList == null || ftVectorList.isEmpty()) {
                        continue;
                    }
                    wordVector = new double[ftVectorList.size()];
                    for (int j = 0; j < ftVectorList.size(); j++) {
                        wordVector[j] = ftVectorList.get(j);
                    }
                    vectorCache.put(token, wordVector);
                }

                // 累加詞向量
                for (int j = 0; j < wordVector.length; j++) {
                    vector[j] += wordVector[j];
                }
                count++;
            }

            // 計算平均詞向量
            if (count > 0) {
                for (int j = 0; j < vector.length; j++) {
                    vector[j] /= count;
                }
            }

            // 建立新實例，最後一個值為類別值
            double[] instanceValues = Arrays.copyOf(vector, vector.length + 1);
            instanceValues[vector.length] = data.instance(i).classValue();
            vectorizedData.add(new DenseInstance(1.0, instanceValues));
        }

        return vectorizedData;
    }

    private static void exportToCSV(Instances data, String csvPath) throws IOException {
        File outputFile = new File(csvPath);

        // 如果檔案已存在，則刪除或生成新檔案名稱
        if (outputFile.exists()) {
            System.out.println("File already exists: " + csvPath);
            if (!outputFile.delete()) {
                System.out.println("**********You may be opening the file, please close it first**********");
                throw new IOException("Unable to delete existing file: " + csvPath);

            }
            System.out.println("Existing file deleted.");
        }

        // 使用 Weka 的 CSVSaver 保存資料
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(outputFile);
        saver.writeBatch();

        System.out.println("Data exported to: " + csvPath);
    }

    public static void saveLabelsToFile(Instances data) {
        try {

            // 文件路徑
            String filePath = "src/main/resources/labels.txt";

            // 寫入類別標籤到文件
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                for (int i = 0; i < data.numClasses(); i++) {
                    String label = data.classAttribute().value(i);
                    writer.write(label);
                    writer.newLine();
                }
            }

            System.out.println("Labels saved to: " + filePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
        private static void initializePipeline() {
        // 使用 CoreNLPHel 來獲取 pipeline
        pipeline = CoreNLPHel.getInstance().getPipeline();
    }

    private static void loadStopWords() {
        try (InputStream inputStream = Train_model.class.getClassLoader().getResourceAsStream("stopwords.txt"); BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stopWords.add(line.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static boolean isStopWord(String word) {
        return stopWords.contains(word);
    }

    private static String preprocessText(String text) {
        // 1. 將全形字元轉換為半形字元
        String normalizedText = Normalizer.normalize(text, Normalizer.Form.NFKC);

        // 2. 使用 Segmentation 進行分詞
        Segmentation segmentation = new Segmentation(normalizedText);
        String tokenizedText = segmentation.getSegtext();

        // 3. 去除停用詞並移除符合正則條件的字詞
        StringTokenizer tokenizer = new StringTokenizer(tokenizedText);
        StringBuilder filteredText = new StringBuilder();

        while (tokenizer.hasMoreTokens()) {
            String word = tokenizer.nextToken();
            // 使用正則表達式過濾掉包含 %, $, _, -, #, +, /, &, (, ), . 的字詞
            if (!isStopWord(word) && !word.matches(".*[%$#\\+/&].*")) {
                filteredText.append(word).append(" ");
            }
        }

        return filteredText.toString().trim();
    }

    private static Instances preprocessDataWithThreads(Instances data) throws Exception {
        ArrayList<String> classValues = new ArrayList<>();
        for (int i = 0; i < data.numClasses(); i++) {
            classValues.add(data.classAttribute().value(i));
        }

        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (List<String>) null));
        attributes.add(new Attribute("class", classValues));
        Instances processedData = new Instances("ProcessedData", attributes, data.numInstances());
        processedData.setClassIndex(processedData.numAttributes() - 1);

        // 建立執行緒池
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<String[]>> futures = new ArrayList<>();

        for (int i = 0; i < data.numInstances(); i++) {
            String originalText = data.instance(i).stringValue(0);

            // 將文本處理任務提交到執行緒池
            futures.add(executor.submit(() -> {
                String processedText = preprocessText(originalText);
                return new String[]{originalText, processedText};
            }));
        }

        // 收集結果
        for (int i = 0; i < futures.size(); i++) {
            try {
                String[] result = futures.get(i).get();
                String originalText = result[0];
                String processedText = result[1];

                if (processedText.isEmpty()) {
                    processedText = "empty";
                }
                double[] values = new double[2];
                values[0] = processedData.attribute(0).addStringValue(processedText);
                values[1] = data.instance(i).classValue();
                processedData.add(new DenseInstance(1.0, values));
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        // 關閉執行緒池
        executor.shutdown();

        return processedData;
    }

    private static JFastText loadFastTextModel(String modelFilePath) throws IOException {
        File modelFile = new File(modelFilePath);
        if (!modelFile.exists()) {
            throw new FileNotFoundException("FastText model file not found: " + modelFilePath);
        }
        JFastText ft = new JFastText();
        ft.loadModel(modelFile.getAbsolutePath());
        System.out.println("FastText model loaded from: " + modelFilePath);
        return ft;
    }


    public static Classifier selectClassifier(String algorithmName) {
        switch (algorithmName.toUpperCase()) {
            case "LR":
                Logistic logistic = new Logistic();
                //logistic.setMaxIts(500);
                //logistic.setRidge(1e-4); // 設置正則化強度
                logistic.setDebug(true); // 啟用 Debug 模式
                return logistic;
            case "NB":
                return new NaiveBayes();
            case "DT":
                return new J48();
            case "RF":
                return new RandomForest();
            case "KNN":
                return new IBk();
            case "SVM":
                return new SMO();
            case "MLP":
                return new MultilayerPerceptron();
            case "XGB":
                return new XGBoostClassifierWrapper();
            default:
                throw new IllegalArgumentException("Unsupported algorithm: " + algorithmName);
        }
    }

    //主程式區塊------------------------------------------------------------------
    public static void main(String[] args) {

        String inputCsvPath = "src/main/resources/inputdata.csv";
        String unifiedModelPath = "src/main/resources/model.model";
        String processedCsvPath = "src/main/resources/processed_data.csv";
        String outputTxtPath = "src/main/resources/data_exploration_results.txt";
        //外部文件須注意
        String fastTextModelPath = "D:/NCU/weka/embedding/fasttext_model_300.bin";

        long startTime = System.currentTimeMillis();
        try {

            // 載入資料
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(inputCsvPath));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            long endTime = System.currentTimeMillis();
            System.out.println("time consuming: " + (endTime - startTime) + " ms");

            // 清空文件內容
            try (PrintWriter writer = new PrintWriter(new File(outputTxtPath))) {
                writer.print("");
            } catch (IOException e) {
                System.out.println("Failed to clear the content of the file: " + e.getMessage());
            }

            // 資料探索並將結果存入檔案
            DataExploration.analyzeClassDistribution(data, outputTxtPath);
            DataExploration.analyzeTextLength(data, outputTxtPath);
            DataExploration.checkMissingValues(data, outputTxtPath);

            // 資料預處理
            Instances processedData = preprocessDataWithThreads(data);

            // 匯出處理後的資料
            exportToCSV(processedData, processedCsvPath);

            fastText = loadFastTextModel(fastTextModelPath);

            // 向量化數據
            Instances vectorizedData = vectorizeData(processedData, fastText);

            // 配置 Resample 過濾器
            Resample resample = new Resample();
            resample.setInputFormat(vectorizedData);
            resample.setNoReplacement(false); // 允許重複
            resample.setBiasToUniformClass(0.5); // 1.0均勻分布、0保持原始分布
            resample.setSampleSizePercent(150); // 增加數據到 150%
            Instances resampledData = Filter.useFilter(vectorizedData, resample);

            // 應用類別平衡過濾器
            ClassBalancer classBalancer = new ClassBalancer();
            classBalancer.setInputFormat(resampledData);
            Instances balancedData = Filter.useFilter(resampledData, classBalancer);

            //模型訓練
            //LR、NB、DT、RF、KNN、SVM、MLP、XGB
            String selectedAlgorithm = "RF";
            Classifier classifier = selectClassifier(selectedAlgorithm);

            //保存類別標籤
            saveLabelsToFile(data);

            // Evaluate model
            Evaluation eval = new Evaluation(balancedData);
            int numFolds = 10; // 10折交叉驗證
            eval.crossValidateModel(classifier, balancedData, numFolds, new Random(6));
            // 輸出評估結果
            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());

            System.out.println("\n=== Evaluation Metrics ===");
            System.out.println("Test Set Accuracy: " + eval.pctCorrect() + "%");
            System.out.println("Precision: " + eval.weightedPrecision());
            System.out.println("Recall: " + eval.weightedRecall());
            System.out.println("F-Measure: " + eval.weightedFMeasure());
            System.out.println("ROC Area: " + eval.weightedAreaUnderROC());

            List<String> classValues = new ArrayList<>();
            for (int i = 0; i < data.numClasses(); i++) {
                classValues.add(data.classAttribute().value(i));
            }
            
            //訓練
            classifier.buildClassifier(balancedData);

            //創建 UnifiedModel 實例
            UnifiedModel unifiedModel = new UnifiedModel(fastText, classifier, classValues);
            // 保存模型，有需要再開啟。
            UnifiedModel.saveModel(unifiedModel, unifiedModelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    //封裝------------------------------------------------------------------------
    //-----------------------------------------------------------------------------
    public static class UnifiedModel implements Serializable {

        private final Classifier classifier;
        // fastText 物件無法序列化，故標記為 transient
        private transient JFastText fastText;
        private final List<String> classValues;
        private final ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();

        public UnifiedModel(JFastText fastText, Classifier classifier, List<String> classValues) {
            this.fastText = fastText;
            this.classifier = classifier;
            this.classValues = classValues;
        }

        public void setFastText(JFastText fastText) {
            this.fastText = fastText;
        }

        private double[] calculateAverageVector(Instances processedData) {
            //int vectorSize = 300;
            int vectorSize = fastText.getVector("示例文本").size(); 
            double[] vector = new double[vectorSize];
            String[] tokens = processedData.instance(0).stringValue(0).split(" ");
            int count = 0;

            for (String token : tokens) {
                double[] wordVector;

                // 檢查緩存中是否已有該詞向量
                if (vectorCache.containsKey(token)) {
                    wordVector = vectorCache.get(token);
                } else {
                    // 使用 fastText.getVector(token) 取得 List<Float>
                    List<Float> ftVectorList = fastText.getVector(token);
                    if (ftVectorList == null || ftVectorList.isEmpty()) {
                        continue;
                    }
                    // 將 List<Float> 轉換成 double[]
                    wordVector = new double[ftVectorList.size()];
                    for (int j = 0; j < ftVectorList.size(); j++) {
                        wordVector[j] = ftVectorList.get(j);
                    }
                    // 加入緩存中
                    vectorCache.put(token, wordVector);
                }

                // 將詞向量累加
                for (int j = 0; j < wordVector.length; j++) {
                    vector[j] += wordVector[j];
                }
                count++;
            }

            // 計算平均值
            if (count > 0) {
                for (int j = 0; j < vector.length; j++) {
                    vector[j] /= count;
                }
            }

            return vector;
        }

        public String predict(String inputText, int quantity) throws Exception {

            // 預處理輸入文本並計算詞向量
            Instances processedData = preprocessData(inputText);
            System.out.println("inputText: " + inputText);
            System.out.println("processedText: " + processedData.instance(0).stringValue(0));
            double[] vector = calculateAverageVector(processedData);

            // 創建預測實例
            Instances instance = createPredictionInstance(vector);

            // 預測結果
            double predictedClassValue = classifier.classifyInstance(instance.instance(0));
            String predictedClass = classValues.get((int) predictedClassValue);

            // 返回 JSON 結果
            Gson gson = new Gson();
            Map<String, Object> jsonResult = new LinkedHashMap<>();
            jsonResult.put("name", inputText);
            jsonResult.put("quantity", quantity);
            jsonResult.put("path", predictedClass);

            return gson.toJson(jsonResult);
        }

        //貼標用方法
        public String predictClass(String inputText) throws Exception {

            Instances processedData = preprocessData(inputText);
            System.out.println("inputText: " + inputText);
            System.out.println("processedText: " + processedData.instance(0).stringValue(0));
            double[] vector = calculateAverageVector(processedData);

            Instances instance = createPredictionInstance(vector);

            double predictedClassValue = classifier.classifyInstance(instance.instance(0));

            // 添加索引範圍檢查
            if (predictedClassValue < 0 || predictedClassValue >= classValues.size()) {
                throw new ArrayIndexOutOfBoundsException("Predicted class value out of range: " + predictedClassValue);
            }

            return classValues.get((int) predictedClassValue);
        }

        private Instances createPredictionInstance(double[] vector) throws Exception {
            ArrayList<Attribute> attributes = new ArrayList<>();
            for (int i = 0; i < vector.length; i++) {
                attributes.add(new Attribute("vec_" + i));
            }
            attributes.add(new Attribute("class", new ArrayList<>(classValues)));

            Instances instance = new Instances("PredictionInstance", attributes, 1);
            instance.setClassIndex(vector.length);
            instance.add(new DenseInstance(1.0, vector));

            return instance;
        }

        private Instances preprocessData(String data) throws Exception {
            // 定義屬性 (文本和類別)
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("text", (List<String>) null));
            attributes.add(new Attribute("class", (List<String>) null));

            Instances processedData = new Instances("ProcessedData", attributes, 1);
            processedData.setClassIndex(processedData.numAttributes() - 1);

            // 分詞和停用詞處理的並行化邏輯
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            List<Future<String>> futures = new ArrayList<>();

            // 將每個文本分詞和停用詞處理分配給執行緒
            String[] sentences = data.split("\\."); // 以句子為單位進行分割處理
            for (String sentence : sentences) {
                futures.add(executor.submit(() -> preprocessText(sentence)));
            }

            List<String> processedSentences = new ArrayList<>();
            try {
                for (Future<String> future : futures) {
                    processedSentences.add(future.get());
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            } finally {
                executor.shutdown();
            }

            // 整合處理後的文本
            StringBuilder processedTextBuilder = new StringBuilder();
            for (String processedSentence : processedSentences) {
                processedTextBuilder.append(processedSentence).append(" ");
            }
            String processedText = processedTextBuilder.toString().trim();

            // 新增至 Instances
            double[] values = new double[2];
            values[0] = processedData.attribute(0).addStringValue(processedText.isEmpty() ? "empty" : processedText);
            values[1] = Utils.missingValue(); // 如果無法定義類別，使用缺失值
            processedData.add(new DenseInstance(1.0, values));

            return processedData;
        }

        public static void saveModel(UnifiedModel model, String filePath) throws IOException {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
                oos.writeObject(model);
                System.out.println("Unified model saved to: " + filePath);
            }
        }

        public static UnifiedModel loadModel(String filePath) throws IOException, ClassNotFoundException {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
                return (UnifiedModel) ois.readObject();
            }
        }

        private static String preprocessText(String text) {
            // 1. 將全形字元轉換為半形字元
            String normalizedText = Normalizer.normalize(text, Normalizer.Form.NFKC);

            // 2. 使用 Segmentation 進行分詞
            Segmentation segmentation = new Segmentation(normalizedText);
            String tokenizedText = segmentation.getSegtext();

            // 3. 去除停用詞並移除符合正則條件的字詞
            StringTokenizer tokenizer = new StringTokenizer(tokenizedText);
            StringBuilder filteredText = new StringBuilder();

            while (tokenizer.hasMoreTokens()) {
                String word = tokenizer.nextToken();
                // 使用正則表達式過濾掉包含 %, $, _, -, #, +, /, &, (, ), . 的字詞
                if (!isStopWord(word) && !word.matches(".*[%$#\\+/&].*")) {
                    filteredText.append(word).append(" ");
                }
            }

            return filteredText.toString().trim();
        }
    }


}
