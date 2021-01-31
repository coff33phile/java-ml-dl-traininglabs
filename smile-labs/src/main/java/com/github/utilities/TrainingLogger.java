package ml.tutorials.utilities;

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
    private static String className;
    private static String trainLogsDir;

    public static Logger loggerSetup(String currentClassName, String trainLogsFolderDirectory) throws IOException {
        className = currentClassName;
        //user specify path
        trainLogsDir = trainLogsFolderDirectory;

        log = logging();

        return log;
    }

    public static Logger loggerSetup(String currentClassName) throws IOException {
        className = currentClassName;
        //default path is Desktop
        trainLogsDir = Paths.get(System.getProperty("user.home"),"\\Desktop\\IntelliJ Training Logs").toString();

        log = logging();

        return log;
    }

    private static Logger logging() throws IOException {
        //setup the logger with class name, later return this
        log = Logger.getLogger(className);

        //get the date for setting up the naming of the folder
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String dateAndTime = format.format(Calendar.getInstance().getTime());

        //get path to create a folder in preferred directory
        Path newFolderPath = Paths.get(trainLogsDir,"\\","TrainingLogs_"+dateAndTime);

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
