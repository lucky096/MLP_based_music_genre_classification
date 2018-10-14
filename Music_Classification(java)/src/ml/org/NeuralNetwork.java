package ml.org;
import java.io.*;
import java.text.*;
import java.util.*;
 
public class NeuralNetwork {
    static {
        Locale.setDefault(Locale.ENGLISH);
    }
 
    
    final DecimalFormat df;
    final Random rand = new Random();
    final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
    final Neuron bias = new Neuron();
    final int[] layers;
    final int randomWeightMultiplier = 1;
    DecimalFormat f=new DecimalFormat("##.##");
    final double epsilon = 0.00000000001;
 
    final double learningRate = 0.9f;
    final double momentum = 0.7f;
    BufferedReader in;
    String str;
    String []ar;
    int i=0;
    final double inputs[][]=new double[400][14];
    final double expectedOutputs[][] =new double [400][14];
    {
 
    // Inputs for xor problem
    try {
        in=new BufferedReader(new FileReader("C:\\Users\\User\\Desktop\\music_classn\\norm_set.txt"));
        while((str=in.readLine())!=null)
    	{
    		ar=str.split("  ");
    		inputs[i][0]=Double.parseDouble(ar[0]);
    		inputs[i][1]=Double.parseDouble(ar[1]);
    		inputs[i][2]=Double.parseDouble(ar[2]);
    		inputs[i][3]=Double.parseDouble(ar[3]);
    		inputs[i][4]=Double.parseDouble(ar[4]);
    		inputs[i][5]=Double.parseDouble(ar[5]);
    		inputs[i][6]=Double.parseDouble(ar[6]);
    		inputs[i][7]=Double.parseDouble(ar[7]);
    		expectedOutputs[i][0]=Double.parseDouble(ar[8]);
    		expectedOutputs[i][1]=Double.parseDouble(ar[9]);
    		expectedOutputs[i][2]=Double.parseDouble(ar[10]);
    		expectedOutputs[i][3]=Double.parseDouble(ar[11]);
    		i++;
    	}
    }
    catch(IOException e)
    {
    	System.out.println(e);
    }

     }
    // Corresponding outputs, xor training data
    
    double resultOutputs[][] =new double[400][4]; // dummy init
    double output[]=new double[400];
    
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(8, 6, 4);
        int maxRuns = 100;
        double minErrorCondition = 0.001;
        nn.run(maxRuns, minErrorCondition);
    }
    
    public NeuralNetwork(int input, int hidden, int output) {
        this.layers = new int[] { input, hidden, output };
        df = new DecimalFormat("#.0#");
 
        /**
         * Create all neurons and connections Connections are created in the
         * neuron class
         */
        for (int i = 0; i < layers.length; i++) {
            if (i == 0) { // input layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    inputLayer.add(neuron);
                }
            } else if (i == 1) { // hidden layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(inputLayer);
                    neuron.addBiasConnection(bias);
                    hiddenLayer.add(neuron);
                }
            }
 
            else if (i == 2) { // output layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(hiddenLayer);
                    neuron.addBiasConnection(bias);
                    outputLayer.add(neuron);
                }
            } else {
                System.out.println("!Error NeuralNetwork init");
            }
        }
 
        // initialize random weights
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
 
        // reset id counters
        Neuron.counter = 0;
        Connection.counter = 0;
 
    }
 
    // random
    double getRandom() {
        return randomWeightMultiplier * (rand.nextDouble() * 2 - 1); // [-1;1[
    }
 
    /**
     * 
     * @param inputs
     *            There is equally many neurons in the input layer as there are
     *            in input variables
     */
    public void setInput(double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }
 
    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }
 
    /**
     * Calculate the output of the neural network based on the input The forward
     * operation
     */
    public void activate() {
        for (Neuron n : hiddenLayer)
            n.calculateOutput();
        for (Neuron n : outputLayer)
            n.calculateOutput();
    }
 
    /**
     * all output propagate back
     * 
     * @param expectedOutput
     *            first calculate the partial derivative of the error with
     *            respect to each of the weight leading into the output neurons
     *            bias is also updated here
     */
    public void applyBackpropagation(double expectedOutput[]) {
 
        // error check, normalize value ]0;1[
        for (int i = 0; i < expectedOutput.length; i++) {
            double d = expectedOutput[i];
            if (d < 0 || d > 1) {
                if (d < 0)
                    expectedOutput[i] = 0 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        }
 
        int i = 0;
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double ak = n.getOutput();
                double ai = con.leftNeuron.getOutput();
                double desiredOutput = expectedOutput[i];
 
                double partialDerivative = -ak * (1 - ak) * ai
                        * (desiredOutput - ak);
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
            i++;
        }
 
        // update weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double aj = n.getOutput();
                double ai = con.leftNeuron.getOutput();
                double sumKoutputs = 0;
                int j = 0;
                for (Neuron out_neu : outputLayer) {
                    double wjk = out_neu.getConnection(n.id).getWeight();
                    double desiredOutput = (double) expectedOutput[j];
                    double ak = out_neu.getOutput();
                    j++;
                    sumKoutputs = sumKoutputs
                            + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
                }
 
                double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }
 
    void run(int maxSteps, double minError) {
        int i;
        // Train neural network until minError reached or maxSteps exceeded
        double error = 1;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            for (int p = 0; p < 100; p++) {
                setInput(inputs[p]);
 
                activate();
 
                output = getOutput();
                resultOutputs[p] = output;
 
                for (int j = 0; j < 4; j++) {
                    double err = Math.pow(Double.parseDouble(f.format(resultOutputs[p][j]))- expectedOutputs[p][j], 2);
                    error += err;
                }
 
                applyBackpropagation(expectedOutputs[p]);
            }
        }
 
        printResult();
         
        System.out.println("Sum of squared errors = " + error);
        //System.out.println("##### EPOCH " + i+"\n");
    }
     
    void printResult()
    {
        System.out.println("Music Classification By Genre");  
        System.out.println("|Duration   |Tempo(BPM)      |  RMS(dB)  | Sampling freq(kHz) | Rate(b)| Range(dB) |Tonality  |No. of Errors |                      Expected Outputs                     |                     Actual Outputs                    |    Class   |");
        for (int p = 0; p < 100; p++) {
            for (int x = 0; x < layers[0]; x++) {
                System.out.printf("%14s",inputs[p][x] + "   ");
            }
            for (int x = 0; x < layers[2]; x++) {
                System.out.printf("%14s",expectedOutputs[p][x] + "   ");
            }
            double max=0;
            int ind=0;
            for (int x = 0; x < layers[2]; x++) {
                System.out.printf("%14s",f.format(resultOutputs[p][x]) + " ");
                if(max<resultOutputs[p][x]){
                	max=resultOutputs[p][x];
                	ind=x;}
            }
            //System.out.print("Class : ");
            if(ind==0)
            {
            	System.out.print("   Classical");
            }
            else if(ind==1)
            {
            	System.out.print("   Rock");
            }
            else if(ind==2)
            {
            	System.out.print("   Jazz");
            }
            else if(ind==3)
            {
            	System.out.print("   Folk");
            }
            System.out.println();
        }
        System.out.println();
    }
}
