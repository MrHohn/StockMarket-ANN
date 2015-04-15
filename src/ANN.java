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
    private int featureLength;
    private NeuralNetwork neuralNet;
    private int inputNum;
    private int newestNetwork;

    public ANN() {
        // Set up the basic parameters for MultiLayerPerceptron
        newestNetwork = 0;
        featureLength = 30;
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

        historicalData = new ArrayList();
        historicalDate = new ArrayList();

        // load the stock data corresponding to the symbol
        loadData(0);

        priceAll = new double[historicalData.size()];
        for (int i = 0; i < historicalData.size(); ++i) {
            priceAll[i] = historicalData.get(i);
        }
        priceMax = max(priceAll);
        System.out.println("max: " + priceMax);

        File netFile = new File("nets/" + SYMBOLS[0] + "-" + historicalDate.get(0) + ".nnet");
        if (!netFile.exists()) {
            TrainingSet trainingSet = new TrainingSet();
            System.out.println(historicalDate.get(0));

            priceTemp = new double[featureLength];
            for (int i = 0; i < featureLength; ++i) {
                priceTemp[i] = historicalData.get(i);
            }
//            priceMax = max(priceTemp);
//            System.out.println("max: " + priceMax);
            for (int i = 0; i < priceTemp.length; ++i) {
                priceTemp[i] = norm(priceTemp[i], priceMax);
            }
            for (int i = 0; i < featureLength - 4; ++i) {
                for (int j = i; j < inputNum; ++j) {
                    inputSet[j] = priceTemp[j];
                }
                trainingSet.addElement(new SupervisedTrainingElement(inputSet, new double[] {priceTemp[i + inputNum]}));
            }

            neuralNet.learnInSameThread(trainingSet);
            neuralNet.save("nets/" + SYMBOLS[0] + "-" + historicalDate.get(0) + ".nnet");
            System.out.println("Finish saving the neural net for " + SYMBOLS[0] + "-" + historicalDate.get(0) + ".");
        }
        else {
            neuralNet = NeuralNetwork.load("nets/" + SYMBOLS[0] + "-" + historicalDate.get(0) + ".nnet");
            System.out.println("Finish loading the neural net for " + SYMBOLS[0] + "-" + historicalDate.get(0) + ".");
        }


        // Start testing
        TrainingSet testSet = new TrainingSet();
        for (int i = 0; i < 5; ++i) {
            for (int j = featureLength - inputNum, k = 0; j < featureLength; ++j, ++k) {
                inputSet[k] = norm(historicalData.get(j), priceMax);
            }
            testSet.addElement(new TrainingElement(inputSet));

            for (TrainingElement testElement : testSet.trainingElements()) {
                neuralNet.setInput(testElement.getInput());
                neuralNet.calculate();
                Vector<Double> networkOutput = neuralNet.getOutput();
                System.out.println("Input is :");
                for (double input : testElement.getInput()) {
                    input = deNorm(input, priceMax);
                    System.out.println(input);
                }
                System.out.println("Output is :");
                for (double output : networkOutput) {
                    output = deNorm(output, priceMax);
                    System.out.println(output);
                }
            }
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
            System.out.println("database operation error 1.");
        }

    }


    public void storeResult() {

        historicalData = new ArrayList();
        historicalDate = new ArrayList();

        try {

            Connection connection;

            connection = DriverManager.getConnection(DatabaseManager.URL + DatabaseManager.DATABASE_NAME, DatabaseManager.USER_NAME, DatabaseManager.PASSWORD);

            Statement statement = connection.createStatement();

            for (int j = 0; j < SYMBOLS.length; j++) {

                loadData(j);

                try {

                    BufferedReader bufferedReader = new BufferedReader(new FileReader("SVM/libsvm-3.20/stock/" + "SVMHistroy_" + SYMBOLS[j] + "_result.txt"));
                    String line;

                    int i = featureLength;

                    while ((line = bufferedReader.readLine()) != null) {
                        statement.executeUpdate("INSERT INTO PredictionSVM VALUES ('" + SYMBOLS[j] + "', '" + historicalDate.get(i) + "', " + Double.parseDouble(line) + ", 'HOLD')");
                        i++;
                    }

                } catch (FileNotFoundException e) {
                    System.out.println("file error!");
                } catch (IOException e) {
                    System.out.println("io error!");
                }

            }

            connection.close();

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("database operation error 2.");
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

    // denorm the number
    private double deNorm(double num, double max) {
        return max * (num - 0.1) / 0.8;
    }


    public static void main(String[] args) {
        ANN ann = new ANN();
        ann.run();
    }
}
