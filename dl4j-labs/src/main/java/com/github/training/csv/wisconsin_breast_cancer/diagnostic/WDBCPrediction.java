package com.github.training.csv.wisconsin_breast_cancer.diagnostic;

import java.io.IOException;

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
        //{
        //ENTER YOUR CODE HERE
        //

        //create a csv record reader and initialise with the filesplit
        //{
        //ENTER YOUR CODE HERE
        //

        //create a list to store the data
        //{
        //ENTER YOUR CODE HERE
        //

        //take each data point and put into the writable list
        //{
        //ENTER YOUR CODE HERE
        //

        //define the schema
//        Schema schema = new Schema.Builder()
//                .addColumnInteger("ID")
//                .addColumnCategorical("diagnosis",Arrays.asList("M","B"))
//                .addColumnDouble("mean-radius")
//                .addColumnDouble("sd-radius")
//                .addColumnDouble("worst-radius")
//                .addColumnDouble("mean-texture")
//                .addColumnDouble("sd-texture")
//                .addColumnDouble("worst-texture")
//                .addColumnDouble("mean-perimeter")
//                .addColumnDouble("sd-perimeter")
//                .addColumnDouble("worst-perimeter")
//                .addColumnDouble("mean-area")
//                .addColumnDouble("sd-area")
//                .addColumnDouble("worst-area")
//                .addColumnDouble("mean-smoothness")
//                .addColumnDouble("sd-smoothness")
//                .addColumnDouble("worst-smoothness")
//                .addColumnDouble("mean-compactness")
//                .addColumnDouble("sd-compactness")
//                .addColumnDouble("worst-compactness")
//                .addColumnDouble("mean-concavity")
//                .addColumnDouble("sd-concavity")
//                .addColumnDouble("worst-concavity")
//                .addColumnDouble("mean-concavepoints")
//                .addColumnDouble("sd-concavepoints")
//                .addColumnDouble("worst-concavepoints")
//                .addColumnDouble("mean-symmetry")
//                .addColumnDouble("sd-symmetry")
//                .addColumnDouble("worst-symmetry")
//                .addColumnDouble("mean-fractaldim")
//                .addColumnDouble("sd-fractaldim")
//                .addColumnDouble("worst-fractaldim")
//                .build();

        //define transform process: remove ID and convert cat to int for diagnosis
        //{
        //ENTER YOUR CODE HERE
        //

        //execute the transform process
        //{
        //ENTER YOUR CODE HERE
        //
//        System.out.println(tp.getInitialSchema());
//        System.out.println(tp.getFinalSchema());

        //read the transformed data using a CollectionRR
        //{
        //ENTER YOUR CODE HERE
        //

        //shuffle
        //{
        //ENTER YOUR CODE HERE
        //

        //split to test and train
        //{
        //ENTER YOUR CODE HERE
        //

        //normalising
        //{
        //ENTER YOUR CODE HERE
        //

        //put the dataset into iterator
        //{
        //ENTER YOUR CODE HERE
        //

        //define config for NN
        //MultiLayerConfiguration config = getNNConfig(seed, lr, nClass, nInput);

        //define config for early stopping
        //arlyStoppingConfiguration earlystopconfig = getEarlyStopConfig(trainIter);

        //set training UI
//        StatsStorage storage = new InMemoryStatsStorage();
//        UIServer server = UIServer.getInstance();
//        server.attach(storage);

        //train the model with earlystoptrainer
        //EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(earlystopconfig,config,trainIter);

//        EarlyStoppingResult res = trainer.fit();



        //evaluation using accuracy, f1, precision and recall




    }



//    private static MultiLayerConfiguration getNNConfig(int seed, double lr, int nClass, int nInput) {
//        //configuration for multilayer network
//        //{
//        //ENTER YOUR CODE HERE
//        //
//
//        //return config;
//    }
//
//    private static EarlyStoppingConfiguration getEarlyStopConfig(DataSetIterator trainIter) {
//            //define stopping condition config
//            //{
//            //ENTER YOUR CODE HERE
//            //
//
//            return earlystopconfig;
//    }

}
