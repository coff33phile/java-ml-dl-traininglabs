package skymind.edugroup.training.csv.heart_disease;

import skymind.edugroup.utilities.TrainingLogger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import smile.classification.Classifier;
import smile.classification.KNN;
import smile.classification.LogisticRegression;
import smile.classification.SVM;
import smile.plot.swing.Canvas;
import smile.plot.swing.ScatterPlot;
import smile.validation.metric.*;

import java.io.File;;
import java.io.IOException;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

public class HeartStudyPrediction {
    private static Logger log;
    private static FileHandler fh = null;

    public static final long seed = 123456;
    private static int batchSize = 64;
    private static int label = 15;
    private static int input = 10;
    private static int hidden = 50;
    private static int output = 2;
    private static double lr = 0.0015;
    private static int epoch = 5;


    public static void main(String[] args) throws Exception, IOException {

        log = TrainingLogger.loggerSetup(HeartStudyPrediction.class.getName(),"C:\\Users\\user\\Desktop\\IntelliJ Training Logs");

        String inputFile = new ClassPathResource("heart/heart-disease-data.csv").getFile().getAbsolutePath();
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(new File(inputFile)));

        //schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("male")
                .addColumnInteger("age")
                .addColumnInteger("education")
                .addColumnInteger("currentSmoker")
                .addColumnInteger("cigsPerDay")
                .addColumnInteger("BPMeds")
                .addColumnInteger("prevalentStroke")
                .addColumnInteger("prevalentHyp")
                .addColumnInteger("diabetes")
                .addColumnInteger("totChol")
                .addColumnDouble("sysBP")
                .addColumnDouble("diaBP")
                .addColumnDouble("BMI") //todouble
                .addColumnInteger("heartRate")
                .addColumnInteger("glucose")
                .addColumnInteger("TenYearCHD")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
//                .filter(new ConditionFilter(new StringColumnCondition("totChol",ConditionOp.Equal,"NA")))
//                .filter(new ConditionFilter(new StringColumnCondition("education",ConditionOp.Equal,"NA")))
//                .filter(new ConditionFilter(new StringColumnCondition("glucose",ConditionOp.Equal,"NA")))
//                .filter(new ConditionFilter(new StringColumnCondition("cigsPerDay",ConditionOp.Equal,"NA")))
//                .filter(new ConditionFilter(new StringColumnCondition("BMI",ConditionOp.Equal,"NA")))
//                .filter(new ConditionFilter(new StringColumnCondition("BPMeds",ConditionOp.Equal,"NA")))
//                .filter(new ConditionFilter(new StringColumnCondition("heartRate",ConditionOp.Equal,"NA")))
//
//                .convertToInteger("totChol")
//                .convertToInteger("education")
//                .convertToInteger("glucose")
//                .convertToInteger("cigsPerDay")
//                .convertToDouble("BMI")
//                .convertToInteger("BPMeds")
//                .convertToInteger("heartRate")

                .filter(new FilterInvalidValues())

                //heartRate also need cleaning, has a NA at line 602 (691 in csv)
                //nullwritable arises from parsing diaBP as integer, it should be double (csv row 17, 46)

                .build();

        List<List<Writable>> originalData = new ArrayList<>();
        while (rr.hasNext()) {
            originalData.add(rr.next()); }

        List<List<Writable>> transformed = LocalTransformExecutor.execute(originalData, tp);
        System.out.println(tp.getInitialSchema());
        System.out.println(tp.getFinalSchema());
        System.out.println(originalData.size());
        System.out.println(transformed.size());

        RecordReader crr = new CollectionRecordReader(transformed);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(crr, transformed.size(), 15,2);

        DataSet dataSet = iter.next();
        dataSet.shuffle();

        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.7);
        DataSet train = splitTestAndTrain.getTrain();
        DataSet test = splitTestAndTrain.getTest();

        // Data normalization
        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(train);
        scaler.transform(train);
        scaler.transform(test);

        INDArray trainFeatures = train.getFeatures();
        INDArray testFeatures = test.getFeatures();
        System.out.println("\nFeatures shape: " + trainFeatures.shapeInfoToString());

        INDArray trainLabels = train.getLabels();

        System.out.println("Labels shape: " + trainFeatures.shapeInfoToString() + "\n");

        double[][] featuresMat = trainFeatures.toDoubleMatrix();

        List<Writable> label_collection = new ArrayList<>();
        for (int i = 0; i < trainLabels.size(0); i++) {
            label_collection.add(transformed.get(i).get(15));
        }

        INDArray label_ind = RecordConverter.toArray(label_collection);
        int[] label_vector = label_ind.toIntVector();

        // Assigning dataset iterator for training purpose
//        ViewIterator trainIter = new ViewIterator(train, batchSize);
//        ViewIterator testIter = new ViewIterator(test, batchSize);
//        train(trainIter, testIter, test);

        KNN<double[]> knn = KNN.fit(featuresMat, label_vector, 3);
        int[] pred = new int[featuresMat.length];
        for (int i = 0; i < featuresMat.length; i++) {
            pred[i] = knn.predict(featuresMat[i]);
        }

        double acc = Accuracy.of(label_vector, pred);
        System.out.println("Accuracy: "+acc);
        double recall = Recall.of(label_vector, pred);
        System.out.println("Recall: "+recall);
        double prec = Precision.of(label_vector, pred);
        System.out.println("Precision: "+prec);
        double f1 = FScore.of(1,label_vector, pred);
        System.out.println("F1: "+f1);

        ConfusionMatrix confusionMat = ConfusionMatrix.of(label_vector, pred);
        System.out.println("Confusion Matrix: "+confusionMat);


//        INDArray predMat = Nd4j.create(pred);
//        INDArray error = label_ind.subi(predMat);
//        Number totalErr = error.sumNumber();
//        System.out.println(totalErr);

        log.info("********************* END ****************************");
    }


    public static void train (DataSetIterator trainIter, DataSetIterator testIter, DataSet test){

        //train model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(lr, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(hidden)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(hidden)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(hidden)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nOut(output)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Initialize UI server for visualization model performance
        log.info("****************************************** UI SERVER **********************************************");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

        // Model training - fit trainIter into model and evaluate model with testIter for each of nEpoch
        log.info("\n*************************************** TRAINING **********************************************\n");
        long timeX = System.currentTimeMillis();
        for (int i = 0; i < epoch; i++) {
            long time = System.currentTimeMillis();
            System.out.println("Epoch" + i + "\n");
            model.fit(trainIter);
            time = System.currentTimeMillis() - time;
            log.info("************************** Done an epoch, TIME TAKEN: " + time + "ms **************************");

            log.info("********************************** VALIDATING *************************************************");
            Evaluation evaluation = model.evaluate(testIter);
            System.out.println(evaluation.stats());
        }
        long timeY = System.currentTimeMillis();

        log.info("\n******************** TOTAL TIME TAKEN: " + (timeY - timeX) + "ms ******************************\n");

        // Print out target values and predicted values
        log.info("\n*************************************** PREDICTION **********************************************");

        INDArray targetLabels = test.getLabels();
        System.out.println("\nTarget shape: " + targetLabels.shapeInfoToString());

        INDArray predictions = model.output(testIter);
        System.out.println("\nPredictions shape: " + predictions.shapeInfoToString() + "\n");

        System.out.println("Target \t\t\t Predicted");

        for (int i = 0; i < 20; i++) {
            System.out.println(targetLabels.getRow(i) + "\t\t" + predictions.getRow(i));
        }

        // Print out model summary
        log.info("\n*************************************** MODEL SUMMARY *******************************************");
        System.out.println(model.summary());

    }
}