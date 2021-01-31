package com.github.solution.csv.abalone;

import com.github.utilities.TrainingLogger;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public class AbaloneRegressionPrediction {

    private static Logger log;

    public static void main(String[] args) throws IOException, InterruptedException {

        log = TrainingLogger.loggerSetup(AbaloneRegressionPrediction.class.getName());

        File filePath = new ClassPathResource("abalone/abalone-data.csv").getFile();
        CSVRecordReader csvrr = new CSVRecordReader(0, ',');
        csvrr.initialize(new FileSplit(filePath));

        Schema schema = getSchema();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .stringToCategorical("sex", Arrays.asList("M", "F", "I"))
                .categoricalToInteger("sex")
                .filter(new FilterInvalidValues())
                .build();

        TransformProcessRecordReader tprr = new TransformProcessRecordReader(csvrr, tp);

        List<List<Writable>> transformed = new ArrayList<>();

        while (tprr.hasNext()) {
            transformed.add(tprr.next());
        }

        log.info("Transformed data size: " + transformed.size());

    }

    private static Schema getSchema() {

        return new Schema.Builder()
                .addColumnString("sex")
                .addColumnsDouble("length", "diameter", "height", "whole-weight",
                        "shucked-weight", "viscera-weight", "shell-weight")
                .addColumnInteger("rings")
                .build();
    }

}
