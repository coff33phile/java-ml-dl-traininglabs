package com.github.training.csv.facebook;

import java.io.IOException;

public class FacebookMetricPrediction {

    static int batchsize = 50;
    static int epoch = 100;
    static int seed = 123;
    static double lr = 0.001;


    public static void main(String[] args) throws IOException, InterruptedException {

        //get file and put into recordreader
        //{
        //INSERT CODE HERE
        //}

        //initialise a nested writable list
//        List<List<Writable>> data = new ArrayList<>();

        //keep the data point one by one into nested writable list
        //{
        //INSERT CODE HERE
        //}

//        System.out.println(data.size());

        //define schema
        //{
        //INSERT CODE HERE
        //}

//        System.out.println(schema.toString());

        //transform process
        //{
        //INSERT CODE HERE
        //}

//        List<List<Writable>> transformed = LocalTransformExecutor.execute(data,tp);
//        System.out.println(tp.getInitialSchema());
//        System.out.println(tp.getFinalSchema());

        //ready using a collection rr
        //{
        //INSERT CODE HERE
        //}

        //shuffle and split into train and test
        //{
        //INSERT CODE HERE
        //}

        //normalise data
        //{
        //INSERT CODE HERE
        //}

        //put into iterator for training
//        ViewIterator trainiter = new ViewIterator(trainset, batchsize);
//        ViewIterator testiter = new ViewIterator(testset, batchsize);

        //get config
//        MultiLayerConfiguration config = getConfig(seed,lr);

        //listeners
//        UIServer server = UIServer.getInstance();
//        StatsStorage storage = new InMemoryStatsStorage();
//        server.attach(storage);

        //init model and train
//        MultiLayerNetwork model = new MultiLayerNetwork(config);
//        model.init();
//        model.setListeners(new ScoreIterationListener(10), new StatsListener(storage,10));
//        model.fit(trainiter,epoch);

        //evaluation
        //{
        //INSERT CODE HERE
        //}


    }

//    private static MultiLayerConfiguration getConfig(int seed, double lr) {
        //model config

//        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
//
//                .build();

//        return config;

//    }

}
