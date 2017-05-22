/*
 * P2 Sistemas Inteligentes. Curso 2015/2016
 * DCCIA. Universidad de Alicante
 */

package p2sineuroph;

import java.io.IOException;



/**
 *
 * @author fidel
 */
public class P2SINeuroph {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        
        //Creo una NN de 2 neuronas en la capa oculta (784 en entrada y 10 salida)
        NN nn = new NN(28*28,25,10);
        
        //Cargo los primeros 10000 valores del train y 100 de test
        nn.loadMNISTData(60000, 600); //normalmente son 70-80% de train y el restante de test
        
        //Entreno la red 
        double learningRate = 0.008;  //este valor varia entre 1 y 0.01
        int maxIterations = 250;
        nn.train(learningRate,maxIterations);
        
        
        
    }    
}
