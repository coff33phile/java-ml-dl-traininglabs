package skymind.edugroup.solution.csv.teaching_assistant;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
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

public class TAEvalPrediction {

    public static void main(String[] args) throws IOException, InterruptedException, IOException {

        int nEpoch = 50;
        int batchsize = 10;
        int seed = 123;
        double lr = 0.001;

        //get the path of the file
        File filepath = new ClassPathResource("teaching_assistant/tae-data.csv").getFile();
        RecordReader csvrr = new CSVRecordReader(0, ',');
        csvrr.initialize(new FileSplit(filepath));

        //define the schema: native, instructor, course, semester, class size, marks-category
        Schema schema = new Schema.Builder()
                .addColumnCategorical("native", Arrays.asList("1","2"))
                .addColumnInteger("instructor")
                .addColumnInteger("course")
                .addColumnInteger("semester")
                .addColumnInteger("class-size")
                .addColumnCategorical("marks-category", "1","2","3")
                .build();

        //define transform process: do one-hot for native
        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToOneHot("native")
                .categoricalToInteger("marks-category")
                .build();

        //create an empty list and put the datapoints into the empty list
        List<List<Writable>> datapoint = new ArrayList<>();
        while (csvrr.hasNext()) {
            datapoint.add(csvrr.next());
        }

        //execute the transform process
        List<List<Writable>> transformed = LocalTransformExecutor.execute(datapoint,tp);
        System.out.println(tp.getInitialSchema());
        System.out.println(tp.getFinalSchema());

        //put dataset into an iterator
        RecordReader crr = new CollectionRecordReader(transformed);
        RecordReaderDataSetIterator datasetiter = new RecordReaderDataSetIterator(crr, transformed.size(), 6, 3);

        //do shuffle and split test train
        DataSet dataset = datasetiter.next();
        dataset.shuffle(123);
        SplitTestAndTrain split = dataset.splitTestAndTrain(0.7);
        DataSet trainset = split.getTrain();
        DataSet testset = split.getTest();

        //do normalising
        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainset);
        scaler.transform(trainset);
        scaler.transform(testset);

        //put the data back into iterator
        ViewIterator trainiter = new ViewIterator(trainset,batchsize);
        ViewIterator testiter = new ViewIterator(testset, batchsize);

        //configuring the NN
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(6)
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(50)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        //declare a server and storage
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //train the model and set listeners
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10),new StatsListener(storage,10));
        for (int i = 0; i < nEpoch; i++) {
            model.fit(trainiter);
        }

        //evaluate using accuracy, f1, precision, recall and confusion matrix
        Evaluation evalTrain = model.evaluate(trainiter);
        System.out.println(evalTrain.stats());
        Evaluation evalTest = model.evaluate(testiter);
        System.out.println(evalTest.stats());

    }

}



