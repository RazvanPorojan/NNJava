//TODO - compare to API
/**
 * @author INSTPorojaG
 *
 */


import java.io.IOException;

import org.json.JSONArray;
import org.json.JSONObject;

import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.types.Cell;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;


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
	private Cell test_data;


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
		//TODO - huge memory waste - the same for nabla_w - weight matrix (tensor?) is bad idea. Should have a weight object made of 2 arrays?
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
			System.out.println(String.format("Loading parameters for Layer %d", i));
			biases[i] = getBiasesFromJSON(biasesS[i-1]);
			weights[i] = getWeightsFromJSON(weightsS[i-1]);
		}
		System.out.println("Done.");
	}


	private double[] getBiasesFromJSON(String biasesS) {
		JSONObject j = new JSONObject("{ \"number\":" + biasesS + "}");
		JSONArray ba = j.getJSONArray("number");
		double[] bi = new double[ba.length()];
		for (int i = 0; i < ba.length(); i++) {		    
			bi[i] = ((JSONArray)ba.get(i)).getDouble(0);
			//System.out.println(String.format("Loaded bias for %d: %f", i, bi[i]));
		}
		return bi;	
	}

	private double[][] getWeightsFromJSON(String weightsS) {
		JSONObject jo = new JSONObject("{ \"number\":" + weightsS + "}");
		JSONArray wa = jo.getJSONArray("number");
		double[][] wi = new double[maxNN][maxNN];
		//System.out.println(String.format("wa.length %d:", wa.length()));
		for (int i = 0; i < wa.length(); i++) {
			JSONArray t = ((JSONArray)wa.get(i));
			//System.out.println(String.format("t.length %d:", t.length()));
			for(int j=0; j < t.length();j++) {				
				wi[i][j] = t.getDouble(j);
				//System.out.println(String.format("Loaded weight for %d,%d: %f", i, j, wi[i][j]));
			}
		}
		return wi;
	}


	public double computeCost(double [] output, double [] label) {
		//TODO
		return maxNN;		
	}

	public void loadDataFromMAT(String fileName) throws IOException {
		Source source = Sources.openFile(fileName);
		test_data = Mat5.newReader(source).readMat().getCell("new_data");

		//Matrix matr = cell.getMatrix(9999);
		//System.out.println(cell);
		//System.out.println(matr);
	}

	public void testWithMATInput(int iterations) {
		//for(int i=0;i<test_data.getNumElements();i++) {
		//System.out.println(test_data.getNumElements());
		int correct_count = 0;
		for(int i=0;i<iterations;i++) {
			Matrix test_item = test_data.getMatrix(i);
			//double[] input = test_item.;
			double[] input_item = MATtoArray(test_item);
			int label = test_data.getMatrix(i,1).getInt(0);
			System.out.println(String.format("Label for training item %d: %d", i, label));

			double[] output = forwardProp(input_item);
			int index = max(output);

			if(index == label) {
				correct_count++;
			}
		}
		System.out.println(String.format("%d correct items from a total of %d", correct_count, iterations));
	}

	private int max(double[] d) {
		double max = 0;
		int j = 0;
		for(int i=0;i<d.length;i++) {
			if(d[i] > max) {
				max = d[i];
				j = i;
			}
		}
		return j;
	}

	private double[] MATtoArray(Matrix i) {
		double[] o = new double[784];//TODO get all cum de a mers cu 768 - pixeli negri?
		for(int j=0;j<784;j++) {
			o[j] = i.getDouble(j);
		}
		return o;
	}

	public double[] forwardProp(double [] input) {
		activations = new double[maxNN][maxNN];
		zs = new double[maxNN][maxNN];
		System.arraycopy(input, 0, activations[0], 0, input.length);
		double z;
		double[] output = null;
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			for(int j=0;j<networkSize[i];j++) {//neuron				
				z = 0;
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer
					z += weights[i][j][k] * activations[i-1][k];
					//System.out.printf("layer %d, neuron %d, link to neuron %d, weight=%f, previous output[%d,%d]=%f, z=%f%n",i,j,k,weights[i][j][k],i-1,k,activations[i-1][k],z);
				}
				z += biases[i][j];
				zs[i][j] = z;
				activations[i][j] = activation(z);
				if (i == numberOfLayers - 1) {
					//System.out.printf("output for neuron %d in layer %d is: %f%n",j,i,activations[i][j]);
				}
			}
		}
		output = activations[numberOfLayers - 1];
		return output;
	}


	/**
	 * Performs Backwards Propagation for a single training example
	 * forwardProp must be called before so we have the outputs stored.
	 */
	public void backProp() {
		int k = numberOfLayers;
		double nabla_b[][] = new double[numberOfLayers][maxNN];
		double nabla_w[][][] = new double[numberOfLayers][maxNN][maxNN];

		//last layer
		double[] delta = hadamard(cost_derivative(activations[k], y), sigmoid_primes(zs[k])); // oki hadamard TODO - repace dot in jupiter with matmul or *
		nabla_b[k] = delta;// - pt. layerul k - fiecare neuron
		nabla_w[k] = multiplyMatrices(arrayToMatrix(delta), arrayToMatrix(activations[k-1], true), 10, 1, 30); //TODO replace hardcoded

		//each other layers
		for(int l=k-1;l>0;l--) {
			//weights 300, delta 30, sig_prime 30 np.dot original tot 30
			delta = hadamard(matrixToArray(multiplyMatrices(arrayToMatrix(weights[l][l-1]), arrayToMatrix(delta, true), 30, 10, 1)), sigmoid_primes(zs[l]));//TODO check if weights[i][i-1] is indeed what we want (vector with weights between layer i and previous layer)
			nabla_b[l] = delta;
			nabla_w[l] = multiplyMatrices(arrayToMatrix(delta), arrayToMatrix(activations[l], true), 10, 1, 30);
		}

	}
	
	private static double[] matrixToArray(double[][] m) {
		return matrixToArray(m, false);
		
	}
	
	private static double[] matrixToArray(double[][] m, boolean transpose) {
		if(transpose) {			
			double[] a = new double[m[0].length];
			System.arraycopy(m[0], 0, a, 0, m[0].length);
			return a;
		}
		double[] a = new double[m.length];
		System.arraycopy(m, 0, a, 0, m.length);
		return a; 
		
	}
	

	private static double[][] arrayToMatrix(double[] a) {
		return arrayToMatrix(a, false);
	}

	private static double[][] arrayToMatrix(double[] a, boolean transpose){
		if(transpose) {
			//System.arraycopy(a, 0, m, 0, a.length); //TODO - care merge
			double[][] m = new double[1][a.length];
			for(int i=0;i<a.length;i++) {
				m[i][0] = a[i];
				return m;
			}
		}
		double[][] m = new double[a.length][1];
		System.arraycopy(a, 0, m[0], 0, a.length);
		return m;

	}

	private static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix, int r1, int c1, int c2) {//r2 = c1
		double[][] product = new double[r1][c2];
		for(int i = 0; i < r1; i++) {
			for (int j = 0; j < c2; j++) {
				for (int k = 0; k < c1; k++) {
					product[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
				}
			}
		}

		return product;
	}


	private double[][] vectProd(double[] x, double[] y) {
		int xl = x.length;
		int yl = y.length;
		double[][] r = new double[xl][yl];

		for(int i=0;i<xl;i++)
			for(int j=0;j<yl;j++)
				r[i][j] = x[i] * y[j];				

		return r;
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
	private static double[] hadamard(double x[], double y[]) {
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
