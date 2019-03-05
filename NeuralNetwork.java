//TODO - compare to API
//TODO - important question do we need lists when we have arrays of arrays as 2D arrays?
//TODO - add suffle to implement real SGD

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
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
	private int numberOfLayers;//including input and output
	private int networkSize[];//number of neurons for each layer including input and output

	private List<double[]> biases;
	private List<double[][]> weights;

	private List<double[]> activations;
	private List<double[]> zs;
	
	//contains sums of gradients in a minibatch
	private List<double[]> nabla_b_sum;
	private List<double[][]> nabla_w_sum;

	private Cell test_data;
	private Cell train_data;

	
	public int[] getNetworkSize() {
		return networkSize;
	}

	public void setNetworkSize(int[] networkSize) {
		this.networkSize = networkSize;
		this.numberOfLayers = networkSize.length;
	}



	public void initialize( ) {
		biases = new ArrayList<double[]>();
		weights = new ArrayList<double[][]>();
		biases.add(null);
		weights.add(null);
		double mean = 0;
		double variance = 1;
		Random r = new Random();
		
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			double[] b = new double[networkSize[i]];
			double[][] w = new double[networkSize[i]][networkSize[i-1]];
			for(int j=0;j<networkSize[i];j++) {//neuron
				
				b[j] = mean + r.nextGaussian() * variance;
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer					
					//System.out.printf("layer %d, neuron %d, link to neuron %d%n",i,j,k);
					w[j][k]= mean + r.nextGaussian() * variance;
				}
			}
			biases.add(i,b);
			weights.add(i,w);
		}
	}
	
	public void modelLoad(String[] weightsS, String[] biasesS ) {
		biases = new ArrayList<double[]>();
		weights = new ArrayList<double[][]>();
		biases.add(null);
		weights.add(null);
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			System.out.println(String.format("Loading parameters for Layer %d", i));
			biases.add(getBiasesFromJSON(biasesS[i-1]));
			weights.add(getWeightsFromJSON(weightsS[i-1],networkSize[i],networkSize[i-1]));
		}
		System.out.println("Done.");
	}

	public double[] forwardProp(double [] input) {
		activations = new ArrayList<double[]>();
		zs = new ArrayList<double[]>();
		double[] a = new double[input.length];
		System.arraycopy(input, 0, a, 0, input.length);//TODO - check if we can add input
		activations.add(a);
		zs.add(null);
		double z;
		double[] output = null;
		for(int i=1;i<numberOfLayers;i++) {//layer - skipping input layer
			double[] actsForLayer = new double[networkSize[i]];
			double[] zsForLayer = new double[networkSize[i]];
			for(int j=0;j<networkSize[i];j++) {//neuron				
				z = 0;
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer					
					z += weights.get(i)[j][k] * activations.get(i-1)[k];
					//System.out.printf("layer %d, neuron %d, link to neuron %d, weight=%f, previous output[%d,%d]=%f, z=%f%n",i,j,k,weights.get(i)[j][k],i-1,k,activations.get(i-1)[k],z);
				}
				z += biases.get(i)[j];
				zsForLayer[j] = z;
				actsForLayer[j] = activation(z);
				if (i == numberOfLayers - 1) {
					//System.out.printf("output for neuron %d in layer %d is: %f%n",j,i,actsForLayer[j]);
				}
			}
			activations.add(actsForLayer);
			zs.add(zsForLayer);
		}
		output =  activations.get(numberOfLayers-1);
		//printArray("output", output);
		return output;
	}
	
	public void trainAll(int epochs, int miniBatchSize, double learningRate, int testSize) {
		for(int i=1;i<=epochs;i++) {
			System.out.println(String.format("Before Epoch %d from %d", i, epochs));
			testWithMATInput(testSize);//TODO - only limited test data			
			trainOnce(miniBatchSize, learningRate);			
		}
		System.out.println(String.format("End of training"));
		testWithMATInput(testSize);//TODO - only limited test data
	}

	private void trainOnce(int miniBatchSize, double learningRate) {

		int numberOfMiniBatches = 50000/miniBatchSize;//TODO fix traindata lenght
		int offset = 0;		
		for(int i=0;i<numberOfMiniBatches;i++) {
			//System.out.println(String.format("Minibatch %d from %d", i, numberOfMiniBatches));
			initNablaSum();
			double miniBatchCost = 0;
			for(int j=0;j<miniBatchSize;j++) {//code below executed for each training item				
				Matrix train_item = train_data.getMatrix(offset);
				double[] input_item = MATtoArray(train_item);
				//printDigit(input_item);
				double[] labelVector = new double[10];//TODO - hardcoded
				int label = 0;
				for(int l=0;l<10;l++) {//TODO - hardcoded
					labelVector[l] = train_data.getMatrix(offset,1).getDouble(l);
					if(labelVector[l]==1) {
						label = l;
					}
				}
				//System.out.println(String.format("Label for training item %d from minibatch %d: %d", j, i, label));
				offset++;
				double[] output = forwardProp(input_item);
				int index = max(output);
				double itemCost = cost(output, labelVector);
				if(index != label) {
					//System.out.println(String.format(" Bad - Label for TRAIN item %d from %d: %d, recognized as:%d - cost:%f", j, miniBatchSize, label, index, itemCost));					
				}
				else {
					//System.out.println(String.format(" OK - Label for TRAIN item %d from %d: %d, recognized as:%d - cost:%f", j, miniBatchSize, label, index, itemCost));	
				}
				miniBatchCost += itemCost;
				backProp(labelVector);		//uses activations[][] field so no output is necessary		
			}
			miniBatchCost = miniBatchCost / miniBatchSize * 100;
			//System.out.printf("Cost for Minibatch %d from %d: %f\r\n", i, numberOfMiniBatches, miniBatchCost);
			
			updateWeights(nabla_b_sum, nabla_w_sum, learningRate, miniBatchSize);
		}
	}
	
	private void initNablaSum() {
		nabla_b_sum = new ArrayList<double[]>();//(re)initialize for each miniBatch
		nabla_w_sum = new ArrayList<double[][]>();
		nabla_b_sum.add(null);
		nabla_w_sum.add(null);
		for(int i=1;i<numberOfLayers;i++) {
			nabla_b_sum.add(new double[networkSize[i]]);
			nabla_w_sum.add(new double[networkSize[i]][networkSize[i-1]]);
		}
	}
	
	public void updateWeights(List<double[]> nabla_b, List<double[][]> nabla_w, double learningRate, int miniBatchSize) {
		for(int i=1;i<numberOfLayers;i++) {
			double[] b = new double[networkSize[i]];
			double[][] w = new double[networkSize[i]][networkSize[i-1]];
			for(int j=0;j<networkSize[i];j++) {//neuron	
				b[j] = biases.get(i)[j];
				b[j] -= learningRate * nabla_b.get(i)[j] / miniBatchSize;
				
				for(int k=0;k<networkSize[i-1];k++) {//neurons in the previous layer
					w[j][k] = weights.get(i)[j][k];
					w[j][k] -= learningRate * nabla_w.get(i)[j][k] / miniBatchSize;
				}				
			}
			biases.set(i, b);
			weights.set(i, w);
		}
	}
	
	/**
	 * Performs Backwards Propagation for a single training example
	 * forwardProp must be called before so we have the outputs stored.
	 */
	public void backProp(double[] labelVector) {
		int k = numberOfLayers - 1;  //minus 1 si update mai jos

		//last layer
		double cD[] = cost_derivative(activations.get(k), labelVector);
		double sP[] = sigmoid_primes(zs.get(k));
		printArray("cD", cD);
		printArray("sP", sP);
		double[] delta = hadamard(cD, sP);
		printArray("delta", sP);
		
		double[] dB = nabla_b_sum.get(k);
		dB = arrayAdd(dB, delta);// - pt. layerul k - fiecare neuron
		printArray("dB", dB);
		nabla_b_sum.set(k, dB);
		
		double[][] dW = nabla_w_sum.get(k);
		dW = matrixAdd(dW,multiplyMatrices(arrayToMatrix(delta), arrayToMatrix(activations.get(k-1), true)));
		nabla_w_sum.set(k, dW);

		//each other layers
		for(int l=k-1;l>0;l--) {
			//weights 300, delta 30, sig_prime 30 np.dot original tot 30
			double[] z = zs.get(l);
			double[] sp = sigmoid_primes(z);
			//delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			delta = matrixToArray(multiplyMatrices(transpose(weights.get(l+1)), arrayToMatrix(delta)));
			delta = hadamard(delta , sp);
			
			double[] ddB = nabla_b_sum.get(l);
			ddB = arrayAdd(ddB, delta);
			nabla_b_sum.set(l, ddB);
			
			double[][] ddW = nabla_w_sum.get(l);
			double[][] nabla_w_p = multiplyMatrices(arrayToMatrix(delta), arrayToMatrix(activations.get(l-1), true));
			ddW = matrixAdd(ddW, nabla_w_p);
			nabla_w_sum.set(l, ddW);
		}

	}

	public void loadTestDataFromMAT(String fileName) throws IOException {//TODO make one method for test and train
		Source source = Sources.openFile(fileName);
		test_data = Mat5.newReader(source).readMat().getCell("new_data");
	}

	public void loadTrainDataFromMAT(String fileName) throws IOException {
		Source source = Sources.openFile(fileName);
		train_data = Mat5.newReader(source).readMat().getCell("new_tr_data");
	}

	public void testWithMATInput(int numberOfTestItems) {		
		int correctCount = 0;
		for(int i=0;i<numberOfTestItems;i++) {
			Matrix test_item = test_data.getMatrix(i);			
			double[] input_item = MATtoArray(test_item);
			int label = test_data.getMatrix(i,1).getInt(0);
			
	
			double[] output = forwardProp(input_item);
			
			int index = max(output);			
			if(index == label) {
				//System.out.println(String.format(" OK - Label for test item %d: %d, recognized as:%d", i, label, index));
				correctCount++;
			}
			else {
				//System.out.println(String.format("BAD - Label for test item %d: %d, recognized as:%d", i, label, index));
			}
		}
		System.out.println(String.format("%d correct items from a total of %d", correctCount, numberOfTestItems));
	}
	

	 private double[][] transpose(double[][] m) {
	        double[][] t = new double[m[0].length][m.length];
	        for (int i = 0; i < m[0].length; i++)
	            for (int j = 0; j < m.length; j++)
	                t[i][j] = m[j][i];
	        return t;
	    }
	 
	private String shape(double[][] m) {
		String s = String.format("(%d,%d)", m.length,m[0].length);
		System.out.println(s);
		return s;
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
	
	//debug method - make it separate
		private void printArray(String name, double[] a) {
			//System.out.println(String.format("%s lenghts %d", name, a.length));
			//for(int i=0;i<a.length;i++){
			//	System.out.println(String.format("%s[%d]=%f", name, i, a[i]));
			//}
			
		}

	

	private double[] MATtoArray(Matrix i) {
		double[] o = new double[784];//TODO get all cum de a mers cu 768 - pixeli negri?
		for(int j=0;j<784;j++) {
			o[j] = i.getDouble(j);
		}
		return o;
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

	private double[][] getWeightsFromJSON(String weightsS, int currentLayerSize, int previousLayerSize) {
		JSONObject jo = new JSONObject("{ \"number\":" + weightsS + "}");
		JSONArray wa = jo.getJSONArray("number");
		double[][] wi = new double[currentLayerSize][previousLayerSize];
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

	public static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {//r2 = c1
		int r1 = firstMatrix.length;
		int c1 = firstMatrix[0].length;
		int c2 = secondMatrix[0].length;
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
}
