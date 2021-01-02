package skymind.edugroup.solution.csv.wisconsin_breast_cancer.diagnostic;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class WDBCPrediction {

    static int nRows = 569;
    static int nEpoch = 1000;
    static int batchsize = 10;
    static int seed = 123;
    static double lr = 0.001;
    static int nClass = 2;
    static int nInput = 30;

    public static void main(String[] args) throws IOException, InterruptedException {

        //get the file path and split it
        File file = new ClassPathResource("wbc/wbc-data.csv").getFile();
        FileSplit fileSplit = new FileSplit(file);

        //create a csv record reader and initialise with the filesplit
        RecordReader csvrr = new CSVRecordReader(1, ',');
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
                .addColumnCategorical("diagnosis",Arrays.asList("M","B"))
                .addColumnDouble("mean-radius")
                .addColumnDouble("sd-radius")
                .addColumnDouble("worst-radius")
                .addColumnDouble("mean-texture")
                .addColumnDouble("sd-texture")
                .addColumnDouble("worst-texture")
                .addColumnDouble("mean-perimeter")
                .addColumnDouble("sd-perimeter")
                .addColumnDouble("worst-perimeter")
                .addColumnDouble("mean-area")
                .addColumnDouble("sd-area")
                .addColumnDouble("worst-area")
                .addColumnDouble("mean-smoothness")
                .addColumnDouble("sd-smoothness")
                .addColumnDouble("worst-smoothness")
                .addColumnDouble("mean-compactness")
                .addColumnDouble("sd-compactness")
                .addColumnDouble("worst-compactness")
                .addColumnDouble("mean-concavity")
                .addColumnDouble("sd-concavity")
                .addColumnDouble("worst-concavity")
                .addColumnDouble("mean-concavepoints")
                .addColumnDouble("sd-concavepoints")
                .addColumnDouble("worst-concavepoints")
                .addColumnDouble("mean-symmetry")
                .addColumnDouble("sd-symmetry")
                .addColumnDouble("worst-symmetry")
                .addColumnDouble("mean-fractaldim")
                .addColumnDouble("sd-fractaldim")
                .addColumnDouble("worst-fractaldim")
                .build();

        //define transform process
        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("ID")
                .categoricalToInteger("diagnosis")
                .build();

        //execute the transform process
        List<List<Writable>> transformedData = LocalTransformExecutor.execute(datalist,tp);
//        System.out.println(tp.getInitialSchema());
//        System.out.println(tp.getFinalSchema());

        //read the transformed data using a CollectionRR
        RecordReader rr = new CollectionRecordReader(transformedData);
        DataSetIterator iter = new RecordReaderDataSetIterator(rr,nRows,0,2);

        //shuffle
        DataSet dataset = iter.next();
        dataset.shuffle();

        //split to test and train
        SplitTestAndTrain splitTestAndTrain = dataset.splitTestAndTrain(0.7);
        DataSet training = splitTestAndTrain.getTrain();
        DataSet test = splitTestAndTrain.getTest();

        //normalising
        DataNormalization normaliser = new NormalizerMinMaxScaler();
        normaliser.fit(training);
        normaliser.transform(training);
        normaliser.transform(test);

        //put the dataset into iterator
        DataSetIterator trainIter = new ViewIterator(training,batchsize);
        DataSetIterator testIter = new ViewIterator(test,batchsize);

        //define config for NN
        MultiLayerConfiguration config = getNNConfig(seed, lr, nClass, nInput);

        //define config for early stopping
        EarlyStoppingConfiguration earlystopconfig = getEarlyStopConfig(trainIter);

        //set training UI
//        StatsStorage storage = new InMemoryStatsStorage();
//        UIServer server = UIServer.getInstance();
//        server.attach(storage);

        //train the model with earlystoptrainer
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(earlystopconfig,config,trainIter);

        EarlyStoppingResult res = trainer.fit();

//        model.setListeners(new ScoreIterationListener(10),new StatsListener(storage,10));
//        model.fit(trainIter,nEpoch);

        //evaluation
//        Evaluation eval = model.evaluate();
//        System.out.println("Accuracy: " + eval.accuracy());
//        System.out.println("F1: " + eval.f1());
//        System.out.println("Confusion matrix: " + eval.confusionMatrix());



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

    private static EarlyStoppingConfiguration getEarlyStopConfig(DataSetIterator trainIter) {
        //define stopping condition config
        EarlyStoppingConfiguration earlystopconfig = new EarlyStoppingConfiguration.Builder<>()
                .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(5))
                .scoreCalculator(new DataSetLossCalculator(trainIter,true))
                .build();

        return earlystopconfig;
    }

}
