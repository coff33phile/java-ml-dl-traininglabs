package skymind.edugroup.utilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class TrainingLogger {

    private static Logger log;

    public static Logger loggerSetup(String className, String trainLogsFolderDirectory) throws IOException {

        //setup the logger with class name, later return this
        log = Logger.getLogger(className);

        //get the date for setting up the naming of the folder
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String dateAndTime = format.format(Calendar.getInstance().getTime());

        //get path to create a folder in preferred directory
        Path newFolderPath = Paths.get(trainLogsFolderDirectory,"\\","TrainingLogs_"+dateAndTime);

        //setup path with folder name
        Path logsPath = Paths.get(newFolderPath.toString(),className+"_"+dateAndTime+".log");

        //create a folder in the directory
        try {
            Files.createDirectories(newFolderPath);
        } catch (IOException dirIO){
            System.err.println("Failed to create directory!"+dirIO.getMessage());
        }

        //setup the logger
        FileHandler fh = new FileHandler(logsPath.toString());
        fh.setFormatter(new SimpleFormatter());
        log.addHandler(fh);

        return log;
    }


}
