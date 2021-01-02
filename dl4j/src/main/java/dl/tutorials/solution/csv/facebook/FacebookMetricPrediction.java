package skymind.edugroup.solution.csv.facebook;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
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

public class FacebookMetricPrediction {

    static int batchsize = 50;
    static int epoch = 100;
    static int seed = 123;
    static double lr = 0.001;


    public static void main(String[] args) throws IOException, InterruptedException {

        //get file and put into recordreader
        File file = new ClassPathResource("facebook/facebook-metrics-dataset.csv").getFile();
        RecordReader csvrr = new CSVRecordReader(1,';');
        csvrr.initialize(new FileSplit(file));

        //initialise a nested writable list
        List<List<Writable>> data = new ArrayList<>();

        //keep the data point one by one into nested writable list
        while (csvrr.hasNext()){
            data.add(csvrr.next());
        }

        System.out.println(data.size());

        //define schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("page-total-likes")
                .addColumnCategorical("type", Arrays.asList("Link","Photo","Status","Video"))
                .addColumnCategorical("category", Arrays.asList("1","2","3"))
                .addColumnCategorical("post-month", Arrays.asList("1","2","3","4","5","6","7","8","9","10","11","12"))
                .addColumnCategorical("post-week-day", Arrays.asList("1","2","3","4","5","6","7"))
                .addColumnInteger("post-hour")
                .addColumnInteger("paid")
                .addColumnInteger("lifetime-post-total-reach")
                .addColumnInteger("lifetime-post-total-impression")
                .addColumnInteger("lifetime-engaged-users")
                .addColumnInteger("lifetime-post-consumers")
                .addColumnInteger("lifetime-post-consumptions")
                .addColumnInteger("lifetime-post-impression-by-fan")
                .addColumnInteger("lifetime-post-reach-by-fan")
                .addColumnInteger("lifetime-post-fan-and-engaged")
                .addColumnInteger("comment")
                .addColumnInteger("like")
                .addColumnInteger("share")
                .addColumnInteger("total-interactions")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToOneHot("type")
                .categoricalToInteger("category")
                .categoricalToInteger("post-month")
                .categoricalToInteger("post-week-day")
                .filter(new ConditionFilter(new CategoricalColumnCondition("paid", ConditionOp.Equal,"?")))
                .filter(new ConditionFilter(new CategoricalColumnCondition("like", ConditionOp.Equal,"?")))
                .filter(new ConditionFilter(new CategoricalColumnCondition("share", ConditionOp.Equal,"?")))
                .build();

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data,tp);
        System.out.println(tp.getInitialSchema());
        System.out.println(tp.getFinalSchema());

        //ready using a collection rr
        RecordReader crr = new CollectionRecordReader(transformed);
        DataSetIterator datasetiter = new RecordReaderDataSetIterator(crr,500,12,12,true);

        //shuffle and split into train and test
        DataSet dataset = datasetiter.next();
        dataset.shuffle();
        SplitTestAndTrain split = dataset.splitTestAndTrain(0.8);
        DataSet trainset = split.getTrain();
        DataSet testset = split.getTest();

        //normalise data
        DataNormalization normaliser = new NormalizerMinMaxScaler();
        normaliser.fitLabel(true);
        normaliser.fit(trainset);
        normaliser.transform(trainset);
        normaliser.transform(testset);

        //put into iterator for training
        ViewIterator trainiter = new ViewIterator(trainset, batchsize);
        ViewIterator testiter = new ViewIterator(testset, batchsize);

        //get config
        MultiLayerConfiguration config = getConfig(seed,lr);

        //listeners
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);

        //init model and train
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10), new StatsListener(storage,10));
        model.fit(trainiter,epoch);

        //evaluation
        RegressionEvaluation eval = model.evaluateRegression(testiter);
        System.out.println("RMSE: " + eval.averagerootMeanSquaredError());


    }

    private static MultiLayerConfiguration getConfig(int seed, double lr) {
        //model config

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(21)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(100)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(100)
                        .nOut(100)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(100)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        return config;

    }


}
