//TODO - compare to API
/**
 * @author INSTPorojaG
 *
 */
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

import org.json.JSONArray;
import org.json.JSONObject;
import org.python.core.PyFile;
import org.python.core.PyList;
import org.python.core.PyTuple;
import org.python.modules.cPickle;

public class NeuralNetwork {
	private int numberOfLayers;//including input ant output
	private int networkSize[];//number of neurons for each layer including input and output
	private double biases[][];//layer (0 for 1st hidden), neuron number in current layer
	private double weights[][][];//layer (0 for 1st hidden), neuron number in current layer, neuron number in previous layer
	//for weights several matrix can work instead of a big one
	private int maxNN;
	private double [][] activations;
	private double [][] zs;
	private double [] y;
	
	
	public int getNumberOfLayers() {
		return numberOfLayers;
	}

	public void setNumberOfLayers(int numberOfLayers) {
		this.numberOfLayers = numberOfLayers;
	}


	public int[] getNetworkSize() {
		return networkSize;
	}

	public void setNetworkSize(int[] networkSize) {
		this.networkSize = networkSize;
	}
	
	public int getMaxNN() {
		return maxNN;
	}

	public void setMaxNN(int maxNN) {
		this.maxNN = maxNN;
	}
	
		
	public void initialize( ) {
		biases = new double[numberOfLayers][maxNN];
		weights = new double[numberOfLayers][maxNN][maxNN];
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			for(int j=0;j<networkSize[i];j++) {//neuron				
				biases[i][j] = 0;
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer					
					//System.out.printf("layer %d, neuron %d, link to neuron %d%n",i,j,k);					
					weights[i][j][k]= (double) Math.random();//use xavier or other - negative? np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
				}
			}
		}
	}
	
	public void modelLoad(String[] weightsS, String[] biasesS ) {
		biases = new double[numberOfLayers][maxNN];
		weights = new double[numberOfLayers][maxNN][maxNN];
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			System.out.println(String.format("Layer %d", i));
			biases[i] = getBiasesFromJSON(biasesS[i-1]);
			weights[i] = getWeightsFromJSON(weightsS[i-1]);
		}
	}
	

	private double[] getBiasesFromJSON(String biasesS) {
		JSONObject j = new JSONObject("{ \"number\":" + biasesS + "}");
		JSONArray ba = j.getJSONArray("number");
		double[] bi = new double[ba.length()];
		for (int i = 0; i < ba.length(); i++) {		    
		    bi[i] = ((JSONArray)ba.get(i)).getDouble(0);
		    System.out.println(String.format("Loaded bias for %d: %f", i, bi[i]));
		}
		return bi;	
	}

	private double[][] getWeightsFromJSON(String weightsS) {
		JSONObject jo = new JSONObject("{ \"number\":" + weightsS + "}");
		JSONArray wa = jo.getJSONArray("number");
		double[][] wi = new double[maxNN][maxNN];
		System.out.println(String.format("wa.length %d:", wa.length()));
		for (int i = 0; i < wa.length(); i++) {
			JSONArray t = ((JSONArray)wa.get(i));
			System.out.println(String.format("t.length %d:", t.length()));
			for(int j=0; j < t.length();j++) {				
				wi[i][j] = t.getDouble(j);
				System.out.println(String.format("Loaded weight for %d,%d: %f", i, j, wi[i][j]));
			}
		}
		return wi;
	}
	
	public void loadInputPickle(String Path) {
		File f = new File(Path);
        InputStream fs = null;
        try {
            fs = new FileInputStream(f);
        } catch (FileNotFoundException e) {
            e.printStackTrace();        
        }
        PyFile pyF = new PyFile(fs);
        PyList pyL = (PyList) cPickle.load(pyF);
        //System.out.println(pyT.__len__());
	}

	public double computeCost(double [] output, double [] label) {
		//TODO
		return maxNN;		
	}

	public void forwardProp(double [] input) {
		activations = new double[maxNN][maxNN];
		zs = new double[maxNN][maxNN];
		System.arraycopy(input, 0, activations[0], 0, input.length);
		double z;
		//output = input;
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			for(int j=0;j<networkSize[i];j++) {//neuron				
				z = 0;
				System.out.printf("New neuron z=%f%n",z);
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer
					z += weights[i][j][k] * activations[i-1][k];
					System.out.printf("layer %d, neuron %d, link to neuron %d, weight=%f, previous output[%d,%d]=%f, z=%f%n",i,j,k,weights[i][j][k],i-1,k,activations[i-1][k],z);
				}
				z += biases[i][j];
				zs[i][j] = z;
				activations[i][j] = activation(z);
				System.out.printf("output for neuron %d in layer %d is: %f%n",j,i,activations[i][j]);
			}
		}
	}
	
	
	/**
	 * Performs Backwards Propagation for a single training example
	 * forwardProp must be called before so we have the outputs stored.
	 */
	public void backProp() {
		int k = numberOfLayers;
		double nabla_b[][] = new double[numberOfLayers][maxNN];
		double nabla_w[][][] = new double[numberOfLayers][maxNN][maxNN];
		
		/*
		 * nu inteleg shape-ul asta
		 * comprehension https://www.i-programmer.info/programming/python/3942-arrays-in-python.html?start=1 - pentru fiecare b si fiecare w mai face o matrice de shape similar
		 * nabla_b = [np.zeros(b.shape) for b in self.biases]
         * nabla_w = [np.zeros(w.shape) for w in self.weights]
		 * 
		 */
		
		//last layer
		double[] delta = hadamard(cost_derivative(activations[k], y), sigmoid_primes(zs[k]));
		nabla_b[k] = delta;// - pt. layerul k - fiecare neuron
		//nabla_w[k] = dot(delta, activations[-2]) //- pt. layerul k - fiecare weight - dot nu e scalar?
		
		//each other layers
		for(int i=k-1;i>0;i--) {
			delta = v_product(dot(weights[i][i-1], delta), sigmoid_primes(zs[i]));//TODO check if weights[i][i-1] is indeed what we want (vector with weights between layer i and previous layer)
			//TODO nabla_b[-l] = delta
            //TODO nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		}
				
	}
	

	//cost derivative - MSE? - vector
	private double[] cost_derivative(double output_activations[], double y[]) {
		double[] c = new double[y.length];
		for(int i=0;i<y.length;i++) {
			c[i]=output_activations[i] - y[i];
		}
		return c;
	}
	
	//scalar
	private double activation(double z) {
		//return z;
		return  sigmoid(z);
	}
	
	//scalar
	private double sigmoid(double z) {
		return 1/(1 + Math.exp(-z));
	}
	
	//scalar
	private double sigmoid_prime(double z) {
		return sigmoid(z)*(1-sigmoid(z));
	}
	
	//vector
	private double[] sigmoid_primes(double[] x) {
		double[] c = new double[x.length];
		for(int i=0;i<x.length;i++) {
			c[i] = sigmoid_prime(x[i]);
		}
		return c;
	}
	
	//hadamard product (term by term) - vector
	private double[] hadamard(double x[], double y[]) {
		double[] c = new double[x.length];
		for(int i=0;i<x.length;i++) {
			c[i] = x[i] * y[i];
		}
		return c;
	}
	//dot product - scalar
	private double dot(double[] x, double[] y) {
		double s = 0;
		for(int i=0;i<x.length;i++) {
			s += x[i] * y[i];
		}
		return s;
	}
	
	//vector
	private double[] v_product(double s, double[] x) {
		double[] c = new double[x.length];
		for(int i=0;i<x.length;i++) {
			c[i] = x[i] * s;
		}
		return c;
	}

}
