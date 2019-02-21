import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;



public class NNTest {

	public static void main(String[] args) throws IOException {
		//testMatrix();
		train();

	}

	public static void testFwProp() throws IOException{
		NeuralNetwork nn = new NeuralNetwork();
		nn.setMaxNN(784);//TODO - waste
		nn.setNumberOfLayers(3);
		//int size[] = {2,20,20,10}; //size of this instead of LayersNumber
		int size[] = {784,30,10}; //size of this instead of LayersNumber
		nn.setNetworkSize(size);
		
		

		nn.modelLoad(getWeights(), getBiases());

		nn.loadTestDataFromMAT("c:/Users/PorojaG/Downloads/new_test_data.mat");
		nn.testWithMATInput(10000);
	}

	private static String[] getBiases() {
		String[] biases= {"[[-1.3400637540436455], [2.238982207302486], [0.06192797004864826], [3.084411564249057], [-0.3910439709896862], [-0.37502996104094527], [0.11926858840400616], [-0.46209943453734675], [-1.087046412955581], [0.3228718042035298], [-0.8152129394609369], [1.694251365843876], [-1.236861754416872], [-0.865535144242077], [-1.430766903759303], [0.2591627763179116], [-2.359384688604487], [-2.3379188517289085], [1.5850933879080984], [-2.3177237907914767], [1.282587521977169], [-0.826457064164213], [-1.8744456572093953], [0.1342310751717119], [1.0583363153606704], [-0.3565332607142802], [0.12432165033721398], [-1.4331315285821575], [-0.1719186747240139], [-3.5949591408126045]]","[[-5.448325549934733], [-0.258219492601037], [-2.073152845250416], [-5.605408704682468], [-2.711969332465192], [-0.9929131487313696], [-1.7671762883454576], [-0.29491654483208735], [-1.2066404847486931], [-2.8539428764546524]]"
		};
		return biases;
	}

	private static String[] getWeights() {
		String w0 = readFile("c:/Users/PorojaG/Downloads/weights0.txt");//, Charset.defaultCharset());
		String w1 = readFile("c:/Users/PorojaG/Downloads/weights1.txt");//, Charset.defaultCharset());
		String[] weights = {w0 ,w1};
		return weights;
	}

	public static void train() throws IOException{
		NeuralNetwork nn = new NeuralNetwork();
		nn.setMaxNN(784);//TODO - waste
		nn.setNumberOfLayers(3);

		int size[] = {784,30,10}; //size of this instead of LayersNumber
		nn.setNetworkSize(size);

		//nn.initialize();
		nn.modelLoad(getWeights(), getBiases());
		nn.loadTrainDataFromMAT("c:/Users/PorojaG/Downloads/new_train_data.mat");
		nn.loadTestDataFromMAT("c:/Users/PorojaG/Downloads/new_test_data.mat");//TODO - avoid null pointer if the data is not loaded
		nn.trainAll(5, 200, 3.0);


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

		double[][] as = nn.multiplyMatrices(aM, aMt, 3, 1, 3);

		System.out.println(as);

		//TODO - test tensor

		double[][][] tens = new double[3][3][3];
		tens = initTens(tens);
		double[][][] tens2 = nn.tensDiv(tens, 2);
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
