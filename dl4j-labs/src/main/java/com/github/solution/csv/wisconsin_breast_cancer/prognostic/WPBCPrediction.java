package com.github.solution.csv.wisconsin_breast_cancer.prognostic;

import com.github.utilities.TrainingLogger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public class WPBCPrediction {

    private static int nEpoch = 50;
    private static int batchSize = 100;
    private static int seed = 123;
    private static double lr = 0.001;
    private static int nClass = 2;
    private static Logger log;

    public static void main(String[] args) throws IOException, InterruptedException {

        log = TrainingLogger.loggerSetup(WPBCPrediction.class.getName());

        //get the file path and split it
        File file = new ClassPathResource("wisconsin_breast_cancer/prognostic/wpbc-data.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);

        //create a csv record reader and initialise with the filesplit
        RecordReader csvrr = new CSVRecordReader(0, ',');
        csvrr.initialize(fileSplit);

        //create a list to store the data
        List<List<Writable>> datalist = new ArrayList<>();

        //take each data point and put into the writable list
        while (csvrr.hasNext()){
            datalist.add(csvrr.next());
        }

        //define the schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("ID")
                .addColumnCategorical("outcome", Arrays.asList("N","R"))
                .addColumnInteger("time")
                .addColumnDouble("mean-radius")
                .addColumnDouble("mean-texture")
                .addColumnDouble("mean-perimeter")
                .addColumnDouble("mean-area")
                .addColumnDouble("mean-smoothness")
                .addColumnDouble("mean-compactness")
                .addColumnDouble("mean-concavity")
                .addColumnDouble("mean-concavepoints")
                .addColumnDouble("mean-symmetry")
                .addColumnDouble("mean-fractaldim")
                .addColumnDouble("se-radius")
                .addColumnDouble("se-texture")
                .addColumnDouble("se-perimeter")
                .addColumnDouble("se-area")
                .addColumnDouble("se-smoothness")
                .addColumnDouble("se-compactness")
                .addColumnDouble("se-concavity")
                .addColumnDouble("se-concavepoints")
                .addColumnDouble("se-symmetry")
                .addColumnDouble("se-fractaldim")
                .addColumnDouble("worst-radius")
                .addColumnDouble("worst-texture")
                .addColumnDouble("worst-perimeter")
                .addColumnDouble("worst-area")
                .addColumnDouble("worst-smoothness")
                .addColumnDouble("worst-compactness")
                .addColumnDouble("worst-concavity")
                .addColumnDouble("worst-concavepoints")
                .addColumnDouble("worst-symmetry")
                .addColumnDouble("worst-fractaldim")
                .addColumnDouble("tumor-size")
                .addColumnDouble("lymph-status")
                .build();

        //define transform process
        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("ID")
                .categoricalToInteger("outcome")
                .filter(new FilterInvalidValues())
                .build();

        //execute the transform process
        List<List<Writable>> transformedData = LocalTransformExecutor.execute(datalist, tp);
        log.info("Initial schema: " + tp.getInitialSchema());
        log.info("Final schema" + tp.getFinalSchema());
        log.info("Initial size: " + datalist.size());
        log.info("Transformed size: " + transformedData.size());


        //read the transformed data using a CollectionRR
        RecordReader rr = new CollectionRecordReader(transformedData);
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, transformedData.size(), 0, 2);

        //shuffle
        DataSet dataset = iter.next();
        dataset.shuffle(seed);

        //split to test and train
        SplitTestAndTrain splitTestAndTrain = dataset.splitTestAndTrain(0.8);
        DataSet training = splitTestAndTrain.getTrain();
        DataSet test = splitTestAndTrain.getTest();

        //normalising
        DataNormalization normaliser = new NormalizerMinMaxScaler();
        normaliser.fit(training);
        normaliser.transform(training);
        normaliser.transform(test);

        //put the dataset into iterator
        DataSetIterator trainIter = new ViewIterator(training, batchSize);
        DataSetIterator testIter = new ViewIterator(test, batchSize);

        //setup training UI
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //setup and init model
        MultiLayerNetwork model = new MultiLayerNetwork(getNNConfig(seed, lr, nClass, trainIter.inputColumns()));
        model.init();
        model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));
        model.fit(trainIter, nEpoch);

        //evaluation
        Evaluation trainEval = model.evaluate(trainIter);
        System.out.println("Test Accuracy: " + trainEval.accuracy());
        System.out.println("Test F1: " + trainEval.f1());
        System.out.println("Test Confusion matrix: \n" + trainEval.confusionMatrix());

        Evaluation testEval = model.evaluate(testIter);
        System.out.println("Test Accuracy: " + testEval.accuracy());
        System.out.println("Test F1: " + testEval.f1());
        System.out.println("Test Confusion matrix: \n" + testEval.confusionMatrix());

        //evaluation metric is not good perhaps due to small data set
        //try repeating this example using kfold

    }

    private static MultiLayerConfiguration getNNConfig(int seed, double lr, int nClass, int nInput) {
        //configuration for multilayer network
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(nInput)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(100)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(nClass)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        return config;
    }

}

