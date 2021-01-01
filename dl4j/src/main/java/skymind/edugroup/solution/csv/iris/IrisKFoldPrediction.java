package skymind.edugroup.solution.csv.iris;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import skymind.edugroup.utilities.TrainingLogger;

import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

public class IrisKFoldPrediction {

    private static MultiLayerNetwork bestModel;

    public static void main(String[] args) throws IOException {

        double lr = 1e-1;
        int epoch = 5;

        Logger log = TrainingLogger.loggerSetup(IrisKFoldPrediction.class.getCanonicalName());

        log.info("Start programme...");

        IrisDataSetIterator irisDataSetIterator = new IrisDataSetIterator();

        DataSet dataSet = irisDataSetIterator.next();
        dataSet.shuffle(1234);
        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.8);
        DataSet trainAndValSet = splitTestAndTrain.getTrain();
        DataSet testSet = splitTestAndTrain.getTest();

        KFoldIterator kFoldIterator = new KFoldIterator(5, trainAndValSet);
        List<Double> valF1List = new ArrayList<>();
        List<Double> trainF1List = new ArrayList<>();

        //for each fold
        int batch = kFoldIterator.batch();
        int currBatch = 0;
        double bestF1 = 0;
        while (kFoldIterator.hasNext()) {

            log.info("Fold: "+currBatch+"/"+batch);
            DataSet trainSet = kFoldIterator.next();
            log.info("No. of training examples in this fold: "+trainSet.numExamples());
            DataSet valSet = kFoldIterator.testFold();
            log.info("No. of validation examples in this fold: "+valSet.numExamples());

            NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
            scaler.fit(trainSet);
            scaler.transform(trainSet);
            scaler.transform(valSet);

            MultiLayerConfiguration conf = getConf(kFoldIterator.inputColumns(), kFoldIterator.totalOutcomes(), lr);

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            for (int i = 0; i < epoch; i++) {

                log.info("Epoch "+epoch+" learning rate: "+model.getLearningRate(1));
                model.fit(trainSet);
            }

            Evaluation evalTrain = model.evaluate(new ViewIterator(trainSet, trainSet.numExamples()));
            log.info("Fold "+currBatch+" train F1\n"+evalTrain.f1());

            Evaluation evalVal = model.evaluate(new ViewIterator(valSet, valSet.numExamples()));
            log.info("Fold "+currBatch+" validation F1\n"+evalVal.f1());

            trainF1List.add(evalTrain.f1());
            valF1List.add(evalVal.f1());

            if (evalVal.f1() > bestF1){
                bestModel = model;
            }
        }
        INDArray trainF1 = Nd4j.create(trainF1List);
        INDArray valF1 = Nd4j.create(valF1List);
        log.info("Training F1 scores for all folds: \n"+trainF1);
        log.info("Validation F1 scores for all folds: \n"+valF1);

        Evaluation evalTest = bestModel.evaluate(new ViewIterator(testSet, testSet.numExamples()));
        log.info("Test set evaluation stats: \n"+evalTest.stats());

    }

    private static MultiLayerConfiguration getConf(int numInput, int numOutput, double lr) {

        HashMap<Integer, Double> schedule = new HashMap<>();
        schedule.put(0, lr);
        schedule.put(2, lr/2);
        schedule.put(4, lr/4);

        return new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Nesterovs(lr, new MapSchedule(ScheduleType.EPOCH, schedule)))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(50)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(50)
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                        .nOut(numOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
    }
}
