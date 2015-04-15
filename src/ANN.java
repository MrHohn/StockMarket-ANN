import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.SupervisedTrainingElement;
import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.LMS;

//import java.text.SimpleDateFormat;
//import java.util.Date;
import java.io.*;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

/**
 * Created by Hon on 4/14/2015.
 */

public class ANN {

    private static final String[] SYMBOLS = {"GOOG", "YHOO", "AAPL", "FB", "MSFT", "AMZN", "SNE", "WMT", "CAJ", "TWTR"};
    private List<Double> historicalData;
    private List<String> historicalDate;
    private int trainingLength;
    private NeuralNetwork neuralNet;
    private int inputNum;
    private int newestNetwork;
    private List<Double> resultData;
    private List<String> resultDate;

    public ANN() {
        // Set up the basic parameters for MultiLayerPerceptron
        newestNetwork = 0;
        trainingLength = 30;
        int maxIterations = 10000;
        inputNum = 4;
        int outputNum = 1;
        int middleLayer = 9;
        neuralNet = new MultiLayerPerceptron(inputNum, middleLayer, outputNum);
        ((LMS) neuralNet.getLearningRule()).setMaxError(0.001);//0-1
        ((LMS) neuralNet.getLearningRule()).setLearningRate(0.7);//0-1
        ((LMS) neuralNet.getLearningRule()).setMaxIterations(maxIterations);//0-1
    }


    public double getTomorrowPrice(){
        if (newestNetwork != 1) {
            return -1;
        }
        return 0;
    }


    public void run() {

        double[] priceTemp;
        double priceMax;
        double[] inputSet = new double[inputNum];
        double[] priceAll;


        for (int symbolOffset = 0; symbolOffset < SYMBOLS.length; ++symbolOffset) {

            historicalData = new ArrayList();
            historicalDate = new ArrayList();
            resultData = new ArrayList();
            resultDate = new ArrayList();

            // load the stock data corresponding to the symbol
            loadData(symbolOffset);

            // calculate the max closing price in history
            priceAll = new double[historicalData.size()];
            for (int i = 0; i < historicalData.size(); ++i) {
                priceAll[i] = historicalData.get(i);
            }
            priceMax = max(priceAll);
//            System.out.println("max: " + priceMax);

            for (int k = 218; k < historicalData.size() - trainingLength; ++k) {

                File netFile = new File("nets/" + SYMBOLS[symbolOffset] + "-" + historicalDate.get(k) + ".nnet");
                if (!netFile.exists()) {
                    System.out.println("Net not exists.");
                    System.out.println("---------- Start training ----------");
                    TrainingSet trainingSet = new TrainingSet();
                    System.out.println("The start date is " + historicalDate.get(k));
                    System.out.println("The end date is " + historicalDate.get(k + trainingLength - 1 - inputNum));

                    priceTemp = new double[trainingLength];
                    for (int i = 0; i < trainingLength; ++i) {
                        priceTemp[i] = historicalData.get(i + k);
                    }
//                priceMax = max(priceTemp);
//                System.out.println("max: " + priceMax);
                    for (int i = 0; i < priceTemp.length; ++i) {
                        priceTemp[i] = norm(priceTemp[i], priceMax);
                    }
                    for (int i = 0; i < trainingLength - inputNum; ++i) {
                        for (int j = i; j < inputNum; ++j) {
                            inputSet[j] = priceTemp[j];
                        }
                        trainingSet.addElement(new SupervisedTrainingElement(inputSet, new double[] {priceTemp[i + inputNum]}));
                    }

                    neuralNet.learnInSameThread(trainingSet);
                    System.out.println("---------- Finish training ----------");
                    neuralNet.save("nets/" + SYMBOLS[symbolOffset] + "-" + historicalDate.get(k) + ".nnet");
                    System.out.println("Successfully save the neural net for " + SYMBOLS[symbolOffset] + "-" + historicalDate.get(k) + ".");
                }
                else {
                    System.out.println("Net exists.");
                    neuralNet = NeuralNetwork.load("nets/" + SYMBOLS[0] + "-" + historicalDate.get(k) + ".nnet");
                    System.out.println("Successfully load the neural net for " + SYMBOLS[symbolOffset] + "-" + historicalDate.get(k) + ".");
                }


                // Start testing
                System.out.println("---------- Start testing ----------");
                TrainingSet testSet = new TrainingSet();

//        // tried to use the same nerual net to predict several days, but the result was not good enough
//        for (int i = 0; i < 5; ++i) {
//            for (int j = trainingLength - inputNum + i, k = 0; j < trainingLength + i; ++j, ++k) {
//                inputSet[k] = norm(historicalData.get(j), priceMax);
//            }
//            testSet.addElement(new TrainingElement(inputSet));
//        }

                for (int j = trainingLength - inputNum, i = 0; j < trainingLength; ++j, ++i) {
                    inputSet[i] = norm(historicalData.get(j + k), priceMax);
                }
                testSet.addElement(new TrainingElement(inputSet));

                for (TrainingElement testElement : testSet.trainingElements()) {
                    neuralNet.setInput(testElement.getInput());
                    neuralNet.calculate();
                    Vector<Double> networkOutput = neuralNet.getOutput();
//            System.out.println("Input is :");
//            for (double input : testElement.getInput()) {
//                input = deNorm(input, priceMax);
//                System.out.println(input);
//            }
                    System.out.println("Output is :");
                    for (double output : networkOutput) {
                        output = Math.round(deNorm(output, priceMax) * 100.0) / 100.0;
                        resultData.add(output);
                        resultDate.add(historicalDate.get(trainingLength + k));
                        System.out.println(historicalDate.get(trainingLength + k) + ": " + output);
                    }
                }

                System.out.println("---------- Finish testing ----------");

            }

            // store the predicted result into database
            storeResult(symbolOffset);

        }

        newestNetwork = 1;

        //Experiments:
        //                   calculated
        //31;3;2009;4084,76 -> 4121 Error=0.01 Rate=0.7 Iterat=100
        //31;3;2009;4084,76 -> 4096 Error=0.01 Rate=0.7 Iterat=1000
        //31;3;2009;4084,76 -> 4093 Error=0.01 Rate=0.7 Iterat=10000
        //31;3;2009;4084,76 -> 4108 Error=0.01 Rate=0.7 Iterat=100000
        //31;3;2009;4084,76 -> 4084 Error=0.001 Rate=0.7 Iterat=10000
    }


    private void loadData(int CompanyIndex) {

        historicalData.clear();
        historicalDate.clear();

        try {

            Connection connection;

            connection = DriverManager.getConnection(DatabaseManager.URL + DatabaseManager.DATABASE_NAME, DatabaseManager.USER_NAME, DatabaseManager.PASSWORD);

            Statement statement = connection.createStatement();

            ResultSet res = statement.executeQuery("SELECT * FROM stockhistorical WHERE Symbol = '" + SYMBOLS[CompanyIndex] + "'");

            while (res.next()) {
                historicalData.add(res.getDouble("ClosePrice"));
                historicalDate.add(res.getString("Date"));
            }

            connection.close();

        } catch (Exception e) {
            System.out.println("database operation error (loading).");
        }

    }


    public void storeResult(int symbolNum) {

        try {

            Connection connection;

            connection = DriverManager.getConnection(DatabaseManager.URL + DatabaseManager.DATABASE_NAME, DatabaseManager.USER_NAME, DatabaseManager.PASSWORD);

            Statement statement = connection.createStatement();

            for (int i = 0; i < resultData.size(); ++i) {
                statement.executeUpdate("INSERT INTO PredictionANN VALUES ('" + SYMBOLS[symbolNum] + "', '" + resultDate.get(i) + "', " + resultData.get(i) + ", 'HOLD')");
            }

            connection.close();

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("database operation error (storing).");
        }

    }


    // get the max input
    private double max(double[] nums) {
        double max = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] > max) {
                max = nums[i];
            }
        }
        return max;
    }


    // normalize the inputs
    private double norm(double num, double max) {
//        double[] norm = new double[nums.length];
//        for (int i =0; i < nums.length; ++i) {
//            norm[i] = (nums[i] / max) * 0.8 + 0.1;
            //  0.8 and 0.1 will be used to avoid the very small (0.0...) and very big (0.9999) values
//        }
//        return norm;
        return (num / max) * 0.8 + 0.1;
    }

    // denormalize the number
    private double deNorm(double num, double max) {
        return max * (num - 0.1) / 0.8;
    }


    public static void main(String[] args) {
        ANN ann = new ANN();
        ann.run();
    }
}
