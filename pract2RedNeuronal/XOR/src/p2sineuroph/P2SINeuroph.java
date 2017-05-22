/*
 * P2 Sistemas Inteligentes. Curso 2015/2016
 * DCCIA. Universidad de Alicante
 */

package p2sineuroph;

import java.io.IOException;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import static org.neuroph.util.TransferFunctionType.SIGMOID;



/**
 *
 * @author fidel
 */
public class P2SINeuroph {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
       
        
        // create new perceptron network
        NeuralNetwork neuralNetwork = new MultiLayerPerceptron(SIGMOID,2, 2, 1);

       // create training set
        DataSet trainingSet = new DataSet(2, 1);

        // add training data to training set (logical OR function)
        trainingSet. addRow (new DataSetRow (new double[]{0, 0},new double[]{0}));
        trainingSet. addRow (new DataSetRow (new double[]{0, 1}, new double[]{1}));
        trainingSet. addRow (new DataSetRow (new double[]{1, 0},new double[]{1}));
        trainingSet. addRow (new DataSetRow (new double[]{1, 1},new double[]{0}));

        // learn the training set
        for(int i = 0; i < 5; i++)
        {
            //get weights of conexions
            Double[] pesos = neuralNetwork.getWeights();

            System.out.println("pesos :");
            for(int j = 0; j < pesos.length; j++)
            {
                System.out.println(pesos[j].toString());
            }

             // calculate network
            neuralNetwork.calculate();

            //calculate the error
            System.out.println("error :");
            for(Double a : neuralNetwork.getOutput())
            {
                System.out.println(a.toString());
            }
            
            neuralNetwork.learn(trainingSet);
        }  
    } 
}
