//TODO - compare to API
/**
 * @author INSTPorojaG
 *
 */


import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

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

	private double nabla_b_sum[][];//contains sums of gradients in a minibatch
	private double nabla_w_sum[][][];//TODO - return hashmap from backprop

	private double [][] activations;
	private double [][] zs;

	private Cell test_data;
	private Cell train_data;


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
					weights[i][j][k]= (double) Math.random();//TODO use xavier or other - negative? np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
				}
			}
		}
	}



	public double computeCost(double [] output, double [] label) {
		//TODO
		return 1.0;		
	}



	public void trainAll(int epochs, int miniBatchSize, double learningRate) {
		for(int i=1;i<=epochs;i++) {
			System.out.println(String.format("Epoch %d from %d", i, epochs));
			testWithMATInput(2000);//TODO - only limited test data
			trainOnce(miniBatchSize, learningRate);			
		}
	}

	private void trainOnce(int miniBatchSize, double learningRate) {

		int numberOfMiniBatches = 50000/miniBatchSize;//TODO fix
		int offset = 0;		
		for(int i=0;i<numberOfMiniBatches;i++) {
			//System.out.println(String.format("Minibatch %d from %d", i, numberOfMiniBatches));
			nabla_b_sum = new double[numberOfLayers][maxNN];//(re)initialize for each miniBatch
			nabla_w_sum = new double[numberOfLayers][maxNN][maxNN];
			double miniBatchCost = 0;
			for(int j=0;j<miniBatchSize;j++) {//code below executed for each training item
				
				Matrix train_item = train_data.getMatrix(offset);
				double[] input_item = MATtoArray(train_item);
				//printDigit(input_item);
				double[] labelVector = new double[10];
				int label = 0;
				for(int l=0;l<10;l++) {
					labelVector[l] = train_data.getMatrix(offset,1).getDouble(l);//TODO - este deja one hot - cum citesc label
					if(labelVector[l]==1) {
						label = l;
					}
				}
				//System.out.println(String.format("Label for training item %d from minibatch %d: %d", j, i, label));
				offset++;
				double[] output = forwardProp(input_item);
				//miniBatchCost += cost(activations[2], labelVector);
				backProp(labelVector);		//uses activations[][] field so no output is necessary		
			}
			//miniBatchCost = miniBatchCost / miniBatchSize * 100;
			//System.out.println(String.format("Cost for Minibatch %d from %d: %f", i, numberOfMiniBatches, miniBatchCost));
			double[][] nabla_b_avg = matDiv(nabla_b_sum, miniBatchSize);
			double[][][] nabla_w_avg = tensDiv(nabla_w_sum, miniBatchSize);
			updateWeights(nabla_b_avg, nabla_w_avg, learningRate);
		}
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
	public void backProp(double[] labelVector) {
		int k = numberOfLayers - 1;

		//last layer
		double[] delta = hadamard(cost_derivative(activations[k], labelVector), sigmoid_primes(zs[k])); // oki hadamard TODO - repace dot in jupiter with matmul or *
		nabla_b_sum[k] = arrayAdd(nabla_b_sum[k], delta);// - pt. layerul k - fiecare neuron
		nabla_w_sum[k] = matrixAdd(nabla_w_sum[k],multiplyMatrices(arrayToMatrix(delta), arrayToMatrix(activations[k-1], true), 10, 1, 30)); //TODO replace hardcoded

		//each other layers
		for(int l=k-1;l>0;l--) {
			//weights 300, delta 30, sig_prime 30 np.dot original tot 30
			delta = hadamard(matrixToArray(multiplyMatrices(arrayToMatrix(weights[l][l-1]), arrayToMatrix(delta, true), 30, 1, 10)), sigmoid_primes(zs[l]));// pusesem 30,10,1 ????  la vectori 1 tb sa fie la mijloc - TODO check if weights[i][i-1] is indeed what we want (vector with weights between layer i and previous layer)
			nabla_b_sum[l] = arrayAdd(nabla_b_sum[l], delta);
			nabla_w_sum[l] = matrixAdd(nabla_w_sum[l],multiplyMatrices(arrayToMatrix(delta), arrayToMatrix(activations[l], true), 10, 1, 30));
		}

	}

	public void updateWeights(double delta_b[][],double delta_w[][][], double learningRate) {
		for(int i=1;i<numberOfLayers;i++) {
			for(int j=0;j<networkSize[i];j++) {//neuron				
				biases[i][j] -= learningRate * delta_b[i][j];
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer
					weights[i][j][k] -= learningRate * delta_w[i][j][k];
				}
			}
		}
	}

	public void testWithMATInput(int numberOfTestItems) {		
		int correctCount = 0;
		for(int i=0;i<numberOfTestItems;i++) {
			Matrix test_item = test_data.getMatrix(i);			
			double[] input_item = MATtoArray(test_item);
			int label = test_data.getMatrix(i,1).getInt(0);
			//System.out.println(String.format("Label for test item %d: %d", i, label));

			double[] output = forwardProp(input_item);
			int index = max(output);

			if(index == label) {
				correctCount++;
			}
		}
		System.out.println(String.format("%d correct items from a total of %d", correctCount, numberOfTestItems));
	}

	private static void shuffleArray(int[] ar)//TODO - not so simple - what we shuffle
	{
		// If running on Java 6 or older, use `new Random()` on RHS here
		Random rnd = ThreadLocalRandom.current();
		for (int i = ar.length - 1; i > 0; i--)
		{
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
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

	private double[] MATtoArray(Matrix i) {
		double[] o = new double[784];//TODO get all cum de a mers cu 768 - pixeli negri?
		for(int j=0;j<784;j++) {
			o[j] = i.getDouble(j);
		}
		return o;
	}

	public void loadTestDataFromMAT(String fileName) throws IOException {//TODO make one method for test and train
		Source source = Sources.openFile(fileName);
		test_data = Mat5.newReader(source).readMat().getCell("new_data");
	}

	public void loadTrainDataFromMAT(String fileName) throws IOException {
		Source source = Sources.openFile(fileName);
		train_data = Mat5.newReader(source).readMat().getCell("new_tr_data");
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


	public static double[] matrixToArray(double[][] m) {
		return matrixToArray(m, false);

	}

	public static double[] matrixToArray(double[][] m, boolean transpose) {
		if(transpose) {			
			double[] a = new double[m[0].length];
			System.arraycopy(m[0], 0, a, 0, m[0].length);
			//for(int i=0;i<m[0].length;i++) { - working alternative
			//	a[i] = m[0][i];				
			//}			
			return a;
		}
		double[] a = new double[m.length];
		//System.arraycopy(m, 0, a, 0, m.length); - bad
		for(int i=0;i<m.length;i++) {
			a[i] = m[i][0];				
		}
		return a;		
	}


	public static double[][] arrayToMatrix(double[] a) {
		return arrayToMatrix(a, false);
	}

	public static double[][] arrayToMatrix(double[] a, boolean transpose){
		if(transpose) {
			//System.arraycopy(a, 0, m, 0, a.length); //TODO - care merge
			double[][] m = new double[1][a.length];
			for(int i=0;i<a.length;i++) {
				m[0][i] = a[i];				
			}
			return m;
		}
		double[][] m = new double[a.length][1];
		//System.arraycopy(a, 0, m[0], 0, a.length);
		for(int i=0;i<a.length;i++) {
			m[i][0] = a[i];
		}
		return m;

	}

	public static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix, int r1, int c1, int c2) {//r2 = c1
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

	//cost derivative - MSE? - vector
	private double[] cost_derivative(double output_activations[], double y[]) {
		double[] c = new double[y.length];
		for(int i=0;i<y.length;i++) {
			c[i]=output_activations[i] - y[i];
			//System.out.println(String.format("i:%d act:%f y:%f act-y:%f",i,output_activations[i],y[i],c[i]));
		}
		return c;
	}
	
	private double cost(double output_activations[], double y[]) {
		double c = 0;
		for(int i=0;i<y.length;i++) {
			double d = output_activations[i] - y[i];
			c += d * d;
			//System.out.println(String.format("i:%d act:%f y:%f act-y:%f",i,output_activations[i],y[i],c[i]));
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

	private static double[] arrayAdd(double x[], double y[]) {//TODO x.length is maxNN - workaround we use y for length
		double[] c = new double[y.length];
		for(int i=0;i<y.length;i++) {
			c[i] = x[i] + y[i];
		}
		return c;
	}

	private static double[][] matrixAdd(double x[][], double y[][]) {//TODO x.length is maxNN - workaround we use y for length
		double[][] c = new double[y.length][y[0].length];
		for(int i=0;i<y.length;i++) {
			for(int j=0;j<y[0].length;j++) {
				c[i][j] = x[i][j] + y[i][j];
			}
		}
		return c;
	}

	public void printDigit(double y[]) {
		int count = 0;
		for(int i=0;i<28;i++) {
			for(int j=0;j<28;j++) {				
				String block = " ";
				int val = (int) (y[count] * 10);
				if(val > 2) block = "░";
				if(val > 4) block = "▒";
				if(val > 6) block = "▓";
				if(val > 8) block = "█";
				System.out.print(block);				
				count++;				
			}
			System.out.println();
		}
	}

	private static double[][] matDiv(double x[][], double divisor) {
		double[][] c = new double[x.length][x[0].length];//TODO x[0] = workaround rapid is not generic
		for(int i=0;i<x.length;i++) {
			for(int j=0;j<x[i].length;j++) {//TODO x[1] = workaround rapid
				c[i][j] = x[i][j] / divisor;
			}
		}
		return c;
	}


	public static double[][][] tensDiv(double x[][][], double divisor) {//TODO x[0] = workaround rapid is not generic
		double[][][] c = new double[x.length][x[0].length][x[0][0].length];
		for(int i=0;i<x.length;i++) {
			for(int j=0;j<x[i].length;j++) {				
				for(int k=0;k<x[i][j].length;k++) {
					c[i][j][k] = x[i][j][k] / divisor;
				}
			}
		}
		return c;
	}

}
