package com.github.solution.csv.abalone;

import com.github.utilities.TrainingLogger;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public class AbaloneRegressionPrediction {

    private static Logger log;

    public static void main(String[] args) throws IOException, InterruptedException {

        log = TrainingLogger.loggerSetup(AbaloneRegressionPrediction.class.getName());

        File filePath = new ClassPathResource("abalone/abalone-data.csv").getFile();
        CSVRecordReader csvrr = new CSVRecordReader(1, ',');
        csvrr.initialize(new FileSplit(filePath));

        Schema schema = getSchema();

        TransformProcess tp = new TransformProcess.Builder(schema)
//                .stringToCategorical("sex", Arrays.asList("M", "F", "I"))
                .categoricalToInteger("sex")
                .filter(new FilterInvalidValues())
                .build();

        TransformProcessRecordReader tprr = new TransformProcessRecordReader(csvrr, tp);

        List<List<Writable>> transformed = new ArrayList<>();

        while (tprr.hasNext()) {
            transformed.add(tprr.next());
        }
        System.out.println(tp.getFinalSchema());

        log.info("Transformed data size: " + transformed.size());

        CollectionRecordReader crr = new CollectionRecordReader(transformed);
        RecordReaderDataSetIterator dataSetIterator = new RecordReaderDataSetIterator(crr, transformed.size(), 8, 8, true);

        DataSet dataSet = dataSetIterator.next();
        SplitTestAndTrain split = dataSet.splitTestAndTrain(0.8);
        DataSet trainSet = split.getTrain();
        DataSet testSet = split.getTest();

        NormalizerMinMaxScaler minMaxScaler = new NormalizerMinMaxScaler();
        minMaxScaler.fitLabel(true);
        minMaxScaler.fit(trainSet);
        minMaxScaler.transform(trainSet);
        minMaxScaler.transform(testSet);

        ViewIterator trainIter = new ViewIterator(trainSet, 100);
        ViewIterator testIter = new ViewIterator(testSet, 100);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
//                .updater(new Nesterovs(0.001, 0.005))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(50)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(50)
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(trainIter.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        log.info(model.summary());

        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new ScoreIterationListener(100), new StatsListener(storage));

        model.fit(trainIter, 5);

        RegressionEvaluation trainEval = model.evaluateRegression(trainIter);
        RegressionEvaluation testEval = model.evaluateRegression(testIter);

        log.info("Train Set Eval:\n" + trainEval.stats());
        log.info("Test Set Eval:\n" + testEval.stats());


    }

    private static Schema getSchema() {

        return new Schema.Builder()
                .addColumnCategorical("sex", Arrays.asList("M", "F", "I"))
                .addColumnsDouble("length", "diameter", "height", "whole-weight",
                        "shucked-weight", "viscera-weight", "shell-weight")
                .addColumnInteger("rings")
                .build();
    }

}
