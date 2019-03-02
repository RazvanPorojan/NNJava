import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;



public class NNTest {

	public static void main(String[] args) throws IOException {
		//testFwProp();
		train();
		//testMatrix();

	}

	public static void testFwProp() throws IOException{
		NeuralNetwork nn = new NeuralNetwork();
		
		int size[] = {784,30,10};
		nn.setNetworkSize(size);
	
		nn.modelLoad(getWeights(), getBiases());

		nn.loadTestDataFromMAT("new_test_data.mat");
		nn.testWithMATInput(1000);
	}

	private static String[] getBiases() {
		String b0 = readFile("biases0.json");
		String b1 = readFile("biases1.json");
		String[] biases = {b0, b1};
		return biases;
	}

	private static String[] getWeights() {
		String w0 = readFile("weights0.json");//, Charset.defaultCharset());
		String w1 = readFile("weights1.json");//, Charset.defaultCharset());
		String[] weights = {w0 ,w1};
		return weights;
	}

	public static void train() throws IOException{
		NeuralNetwork nn = new NeuralNetwork();

		int size[] = {784,30,10};
		nn.setNetworkSize(size);

		nn.initialize();
		//nn.modelLoad(getWeights(), getBiases());
		nn.loadTrainDataFromMAT("new_train_data.mat");
		nn.loadTestDataFromMAT("new_test_data.mat");//TODO - avoid null pointer if the data is not loaded
		nn.trainAll(50, 10, 3, 10000);


		System.out.flush();
	}

	//@Test
	public static void testMatrix() {
		NeuralNetwork nn = new NeuralNetwork();
		double[] a = {1.0, 2.0, 3.0};
		double[][] aM = nn.arrayToMatrix(a);
		double[] b = nn.matrixToArray(aM);


		System.out.println(b);

		double[][] aMt = nn.arrayToMatrix(a, true);
		double[] bt = nn.matrixToArray(aMt, true);

		System.out.println(bt);

		double[][] as = nn.multiplyMatrices(aM, aMt);

		System.out.println(as);

		//TODO - test tensor

		double[][][] tens = new double[3][3][3];
		tens = initTens(tens);
		//double[][][] tens2 = nn.tensDiv(tens, 2);
		System.out.println(tens);

	}

	private static double[][][] initTens(double x[][][]) {
		double z = 2.0;
		double[][][] c = new double[x.length][x[0].length][x[0][0].length];
		for(int i=0;i<x.length;i++) {
			for(int j=0;j<x[0].length;j++) {				
				for(int k=0;k<x[0][0].length;k++) {
					c[i][j][k] = z;
					z += 2;
				}
			}
		}
		return c;
	}
	static String readFile(String path, Charset encoding) 
			throws IOException 
	{
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}

	private static String readFile(String filePath) 
	{
		String content = "";
		try 
		{
			content = new String ( Files.readAllBytes( Paths.get(filePath) ) );
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		return content;
	}

}
