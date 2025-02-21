package com.mycompany.train_model;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class CoreNLPHel {

    private static CoreNLPHel instance = new CoreNLPHel();
    private StanfordCoreNLP pipeline;

    private CoreNLPHel() {
        String props = "CoreNLP-chinese.properties";  //第三步骤的配置文件，放在main/resources目录下
        pipeline = new StanfordCoreNLP(props);
    }

    ;
    public static CoreNLPHel getInstance() {
        return instance;
    }

    public StanfordCoreNLP getPipeline() {
        return pipeline;
    }
}
