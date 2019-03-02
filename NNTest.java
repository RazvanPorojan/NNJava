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
		nn.loadTestDataFromMAT("new_test_data.mat");
		nn.trainAll(50, 10, 3, 10000);


		System.out.flush();
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
