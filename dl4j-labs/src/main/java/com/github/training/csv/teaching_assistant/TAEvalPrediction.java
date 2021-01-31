package com.github.training.csv.teaching_assistant;

import java.io.IOException;

public class TAEvalPrediction {
    
        public static void main(String[] args) throws IOException, InterruptedException {

            int nEpoch = 50;
            int features = 5;
            int target = 1;
            int batchsize = 10;
            int seed = 123;
            double lr = 0.001;

            //get the path of the file
            //{
            //ENTER YOUR CODE HERE
            //}

            //define the schema: native, instructor, course, semester, class size, marks-category
            //{
            //ENTER YOUR CODE HERE
            //}

            //define transform process: do one-hot for native and semester
            //{
            //ENTER YOUR CODE HERE
            //}

            //create an empty list and put the datapoints into the empty list
            //{
            //ENTER YOUR CODE HERE
            //}

            //execute the transform process
            //{
            //ENTER YOUR CODE HERE
            //}
//            System.out.println(tp.getInitialSchema());
//            System.out.println(tp.getFinalSchema());

            //put dataset into an iterator
            //{
            //ENTER YOUR CODE HERE
            //}

            //do shuffle and split test train
            //{
            //ENTER YOUR CODE HERE
            //}

            //do normalising
            //{
            //ENTER YOUR CODE HERE
            //}

            //put the data back into iterator
            //{
            //ENTER YOUR CODE HERE
            //}

            //configuring the NN
            //{
            //ENTER YOUR CODE HERE
            //}

            //declare a server and storage
//            StatsStorage storage = new InMemoryStatsStorage();
//            UIServer server = UIServer.getInstance();
//            server.attach(storage);

            //train the model and set listeners
            //{
            //ENTER YOUR CODE HERE
            //}

            //evaluate using accuracy, f1, precision, recall and confusion matrix
            //{
            //ENTER YOUR CODE HERE
            //}




        }

}
