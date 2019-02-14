import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

public class NNTest {

	public static void main(String[] args) throws IOException {
		NeuralNetwork nn = new NeuralNetwork();
		nn.setMaxNN(784);//TODO - waste
		nn.setNumberOfLayers(3);
		//int size[] = {2,20,20,10}; //size of this instead of LayersNumber
		int size[] = {784,30,10}; //size of this instead of LayersNumber
		nn.setNetworkSize(size);
		String[] biases= {"[[-1.3400637540436455], [2.238982207302486], [0.06192797004864826], [3.084411564249057], [-0.3910439709896862], [-0.37502996104094527], [0.11926858840400616], [-0.46209943453734675], [-1.087046412955581], [0.3228718042035298], [-0.8152129394609369], [1.694251365843876], [-1.236861754416872], [-0.865535144242077], [-1.430766903759303], [0.2591627763179116], [-2.359384688604487], [-2.3379188517289085], [1.5850933879080984], [-2.3177237907914767], [1.282587521977169], [-0.826457064164213], [-1.8744456572093953], [0.1342310751717119], [1.0583363153606704], [-0.3565332607142802], [0.12432165033721398], [-1.4331315285821575], [-0.1719186747240139], [-3.5949591408126045]]","[[-5.448325549934733], [-0.258219492601037], [-2.073152845250416], [-5.605408704682468], [-2.711969332465192], [-0.9929131487313696], [-1.7671762883454576], [-0.29491654483208735], [-1.2066404847486931], [-2.8539428764546524]]"
};
		String w0 = readFile("c:/Users/PorojaG/Downloads/weights0.txt");//, Charset.defaultCharset());
		String w1 = readFile("c:/Users/PorojaG/Downloads/weights1.txt");//, Charset.defaultCharset());
		
		//System.out.println(readFileS());
		String[] weights = {w0 ,w1};
		nn.modelLoad(weights, biases);
		
		nn.loadDataFromMAT("c:/Users/PorojaG/Downloads/new_test_data.mat");
		nn.testWithMATInput(10000);
		
		
		//nn.initialize();
		//double input[] = {1,1};
		//nn.forwardProp(input);
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
