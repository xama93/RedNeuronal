/*
 * P2 Sistemas Inteligentes. Curso 2015/2016
 * DCCIA. Universidad de Alicante
 */

package p2sineuroph;

import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.samples.convolution.MNISTDataSet;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author Fidel
 */
public class NN implements LearningEventListener {
    
    public NN(int inputSize, int hiddenLayerSize, int outputSize){
       _mlp = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputSize, hiddenLayerSize, outputSize);
       _bp = new BackPropagation();
       
       //Para recibir el evento en cada iteración de BP
       _bp.addListener(this);
    }
    
    
    public void loadMNISTData(int trainSamples, int testSamples) {
        
        System.out.println("Loading MNIST data...");
        
        if(trainSamples>0 && trainSamples<=60000 && testSamples>0 && testSamples<=10000){
            try {
            _trainSet = MNISTDataSet.createFromFile(MNISTDataSet.TRAIN_LABEL_NAME, MNISTDataSet.TRAIN_IMAGE_NAME, trainSamples);
            _testSet = MNISTDataSet.createFromFile(MNISTDataSet.TEST_LABEL_NAME, MNISTDataSet.TEST_IMAGE_NAME, testSamples);
            } catch (IOException ex) {
                Logger.getLogger(NN.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            System.err.println("Error in loadMNISTData: fuera de rango. trainSamples debe estar entre 1 y 60000; testSamples entre 1 y 10000");
        }

    }
    
    public void randomize(){
        _mlp.randomizeWeights(); 
    }
    
    public void train(double learningRate, int maxIterations) {
        
        System.out.println("Training network...");
        
        _bp.setMaxIterations(maxIterations);
        _bp.setLearningRate(learningRate);
        _mlp.learn(_trainSet,_bp);
        
    }
    
    public int compararsalida(double[] salida){
        
        double mayor = 0;
        int pos = 0;
        
        for(int i = 0; i < salida.length; i++)
        {
            if(salida[i] > mayor)
            {
                mayor = salida[i];
                pos = i;
            }
        }
        return pos;
    }
    
    int aciertos;
    int fallos;
    
    public double testAccuracy(DataSet set)
    {  
        double media = 0;
        aciertos = 0;
        fallos = 0;
        
        for(int i = 0; i < 1000; i++)
        {
            DataSetRow imagen= set.getRowAt(i);
            
            _mlp.setInput(imagen.getInput());
            _mlp.calculate();
            
            double[] sal = _mlp.getOutput();
            
            if(compararsalida(imagen.getDesiredOutput()) == compararsalida(sal))
            {
                aciertos++;
                
            }else
            {    
                fallos++;
            }
        }
        
        media = aciertos/(double)(aciertos + fallos);
        
        return media;
    }
    
    @Override
    public void handleLearningEvent(LearningEvent le) {
        BackPropagation bp = (BackPropagation) le.getSource();
        System.out.println(bp.getCurrentIteration() + ") Network error: " + bp.getTotalNetworkError());
        double media = testAccuracy(_trainSet);
        System.out.println("Se han obtenido " + aciertos + " aciertos y " + fallos + " fallos, la media es " + media + ";");
    }
    
    public DataSet _trainSet;
    public DataSet _testSet;
    public MultiLayerPerceptron _mlp;
    public BackPropagation _bp;
}
