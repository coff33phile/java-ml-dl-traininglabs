package dl.tutorials.solution.csv.wisconsin_breast_cancer.original;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
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
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import skymind.edugroup.utilities.TrainingLogger;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class WBCOriginalPrediction {

    private static Logger log;

    public static void main(String[] args) throws IOException, InterruptedException {

        log = TrainingLogger.loggerSetup(WBCOriginalPrediction.class.getName());

        //get the file
        File filePath = new ClassPathResource("wisconsin_breast_cancer/original/wbc-original.csv").getFile();
        CSVRecordReader csvrr = new CSVRecordReader(0, ',');
        csvrr.initialize(new FileSplit(filePath));

        //return schema
        Schema schema = getSchema();

        //define transform process
        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("sample code number")
                .filter(new FilterInvalidValues())
                //change malignant to 1
                .conditionalReplaceValueTransform("class", new IntWritable(1),
                        new IntegerColumnCondition("class", ConditionOp.Equal, 4))
                //change benign to 0
                .conditionalReplaceValueTransform("class", new IntWritable(0),
                        new IntegerColumnCondition("class", ConditionOp.Equal, 2))
                .build();

        //define a method to perform data transformation on the fly
        TransformProcessRecordReader tprr = new TransformProcessRecordReader(csvrr, tp);

        List<List<Writable>> original = new ArrayList<>();
        List<List<Writable>> transformed = new ArrayList<>();

        while (csvrr.hasNext()){
            original.add(csvrr.next());
        }
        csvrr.reset();
        while (tprr.hasNext()){
            transformed.add(tprr.nextRecord().getRecord());
        }

        log.info("Original size: "+original.size());
        log.info("Transformed size: "+transformed.size());

        CollectionRecordReader crr = new CollectionRecordReader(transformed);
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(crr, transformed.size(), 9, 2);

        //perform splitting, take .8 training split
        DataSet dataSet = dataIter.next();
        SplitTestAndTrain splitTestTrain = dataSet.splitTestAndTrain(0.8);
        DataSet trainSet = splitTestTrain.getTrain();
        DataSet testSet = splitTestTrain.getTest();

        //define and perform scaling
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainSet);
        scaler.transform(trainSet);
        scaler.transform(testSet);

        ViewIterator trainIter = new ViewIterator(trainSet, 100);
        ViewIterator testIter = new ViewIterator(testSet, 100);

        //return the model config
        MultiLayerConfiguration config = getConfig(123, trainIter.inputColumns(), trainIter.totalOutcomes(), 1e-2);

        //initialise the model weights
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        log.info(model.summary());

        //define listeners
        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(10));

        //train the model
        int epoch = 5;
        model.fit(trainIter, epoch);

        //evaluate the model
        Evaluation evalTrain = model.evaluate(trainIter);
        log.info(evalTrain.stats());
        Evaluation evalTest = model.evaluate(testIter);
        log.info(evalTest.stats());

    }

    private static Schema getSchema() {
        //input data schema
        return new Schema.Builder()
                .addColumnsInteger(
                        "sample code number",
                        "clump thickness",
                        "cell size uniformity",
                        "cell shape uniformity",
                        "marginal adhesion",
                        "single epithelial cell size",
                        "bare nuclei",
                        "bland chromatin",
                        "normal nucleoli",
                        "mitoses",
                        "class")

                .build();
    }

    private static MultiLayerConfiguration getConfig(int seed, int inputNum, int outputNum, double lr) {
        //writing model config
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Nesterovs(lr, new StepSchedule(ScheduleType.ITERATION, lr, 1e-4, 1)))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputNum)
                        .nOut(50)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nOut(outputNum)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
    }

}
