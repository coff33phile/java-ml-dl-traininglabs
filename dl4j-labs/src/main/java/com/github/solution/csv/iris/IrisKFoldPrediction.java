package com.github.solution.csv.iris;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import com.github.utilities.TrainingLogger;

import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

public class IrisKFoldPrediction {

    private static MultiLayerNetwork bestModel;

    public static void main(String[] args) throws IOException {

        double lr = 1e-1;
        int epoch = 10;
        int numFold = 5;

        //setup the training logger
        Logger log = TrainingLogger.loggerSetup(IrisKFoldPrediction.class.getCanonicalName());

        log.info("Start programme...");

        //get the iris dataset
        IrisDataSetIterator irisDataSetIterator = new IrisDataSetIterator();

        //perform shuffling and split into train, validation and test
        DataSet dataSet = irisDataSetIterator.next();
        dataSet.shuffle(1234);
        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.8);
        DataSet trainAndValSet = splitTestAndTrain.getTrain();
        DataSet testSet = splitTestAndTrain.getTest();

        //perform Kfold on train and validation set
        KFoldIterator kFoldIterator = new KFoldIterator(numFold, trainAndValSet);
        //create two list to keep F1 scores
        List<Double> valF1List = new ArrayList<>();
        List<Double> trainF1List = new ArrayList<>();

        //for each fold
        int currBatch = 0;
        double bestF1 = 0;
        while (kFoldIterator.hasNext()) {
            log.info("Fold: " + currBatch + "/"+numFold);
            //get train set for this current fold
            DataSet trainSet = kFoldIterator.next();
            log.info("No. of training examples in this fold: " + trainSet.numExamples());
            //get validation set for this current fold
            DataSet valSet = kFoldIterator.testFold();
            log.info("No. of validation examples in this fold: " + valSet.numExamples());

            //perform normalisation
            NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
            scaler.fit(trainSet);
            scaler.transform(trainSet);
            scaler.transform(valSet);

            //return model config
            MultiLayerConfiguration conf = getConf(kFoldIterator.inputColumns(), kFoldIterator.totalOutcomes(), lr);

            //init model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.summary();
            model.setListeners(new ScoreIterationListener(1));

            //training
            for (int i = 0; i < epoch; i++) {
                model.fit(new ViewIterator(trainSet, trainSet.numExamples()));
                //must supply an iterator if want to use getEpochCount() and getLearningRate()
                log.info("Epoch " + model.getEpochCount() + " learning rate: " + model.getLearningRate(0));
            }

            //calculate training and validation F1 score for current fold
            Evaluation evalTrain = model.evaluate(new ViewIterator(trainSet, trainSet.numExamples()));
            log.info("Fold " + currBatch +" train F1\n" + evalTrain.f1());

            Evaluation evalVal = model.evaluate(new ViewIterator(valSet, valSet.numExamples()));
            log.info("Fold " + currBatch + " validation F1\n" + evalVal.f1());

            //add the F1 score into the list
            trainF1List.add(evalTrain.f1());
            valF1List.add(evalVal.f1());

            //save the model as bestModel if the F1 score is better than the current best
            if (evalVal.f1() > bestF1){
                bestModel = model;
            }

            //update counter
            currBatch = currBatch + 1;
        }

        //print out F1 score for each fold
        INDArray trainF1 = Nd4j.create(trainF1List);
        INDArray valF1 = Nd4j.create(valF1List);
        log.info("Training F1 scores for all folds: \n" + trainF1);
        log.info("Validation F1 scores for all folds: \n" + valF1);

        //evaluate the best model on test set
        Evaluation evalTest = bestModel.evaluate(new ViewIterator(testSet, testSet.numExamples()));
        log.info("Test set evaluation stats: \n" + evalTest.stats());

    }

    private static MultiLayerConfiguration getConf(int numInput, int numOutput, double lr) {

        //write a learning rate scheduler
        HashMap<Integer, Double> schedule = new HashMap<>();
        schedule.put(0, lr);
        schedule.put(2, lr/2);
        schedule.put(4, lr/4);
        schedule.put(6, lr/8);
        schedule.put(7, lr/16);
        schedule.put(8, lr/32);

        //return a network config
        return new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, schedule)))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(50)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(100)
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nOut(numOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
    }
}
